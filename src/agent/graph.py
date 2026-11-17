"""LangGraph state machine orchestration for autonomous incident triage."""

import json
import re
import time
import unicodedata
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from src.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from src.agent.tools import (
    apply_resolution,
    get_order_details,
    search_compensation_policy,
)
from src.config import Settings, get_settings
from src.schemas import (
    ActionType,
    DecisionOutput,
    IncidentCategory,
    IncidentResolveRequest,
    IncidentResolveResponse,
    ToolCallLog,
)
from src.utils.telemetry import (
    TelemetryCollector,
    estimate_tokens,
    get_logger,
)

_LOGGER = get_logger("ifood_triage_agent.graph")

_INJECTION_PATTERNS = (
    r"ignore\s+(todas\s+)?(as\s+)?(suas\s+)?regras",
    r"ignore\s+(all\s+)?(previous\s+)?instructions",
    r"desconsidere\s+(as\s+)?(suas\s+)?(regras|instrucoes)",
    r"system\s*prompt",
    r"modo\s+desenvolvedor",
    r"developer\s+mode",
    r"jailbreak",
    r"bypass\s+(guardrail|policy|regras)",
    r"esquece\s+as\s+regras",
)

_CURRENCY_PATTERNS = (
    r"r\$\s*(\d+(?:[.,]\d{1,2})?)",
    r"(?:cupom|reembolso|estorno|credito)\s+(?:de|no\s+valor\s+de)?\s*(?:r\$\s*)?(\d+(?:[.,]\d{1,2})?)",
    r"(\d+(?:[.,]\d{1,2})?)\s*reais",
)


class LLMDecisionProposal(BaseModel):
    """Structured reasoning output returned by LLM or deterministic policy engine."""

    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    amount: float = Field(default=0.0, ge=0.0)
    justification: str = Field(..., min_length=1)
    policy_reference: str
    customer_response_draft: str = Field(..., min_length=1)


class IncidentGraphState(TypedDict):
    """Explicit state carried across all LangGraph triage nodes."""

    trace_id: str
    order_id: str
    customer_message: str
    incident_category: IncidentCategory
    prompt_injection_detected: bool
    extracted_injected_amount: Optional[float]
    order_result: dict[str, Any]
    policy_text: str
    proposed_action: ActionType
    proposed_amount: float
    proposed_justification: str
    policy_reference: str
    customer_response_draft: str
    resolution_output: dict[str, Any]
    decision: dict[str, Any]
    customer_response: str
    tools_executed: list[dict[str, Any]]
    state_transitions: list[str]
    input_tokens: int
    output_tokens: int
    provider_used: str
    model_used: str


def _normalize(text: str) -> str:
    """Lowercase and remove accents for deterministic pattern matching."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _extract_monetary_amount(normalized_text: str) -> Optional[float]:
    """Extract the largest monetary amount explicitly requested in normalized text."""
    amounts: list[float] = []
    for pattern in _CURRENCY_PATTERNS:
        for match in re.finditer(pattern, normalized_text):
            raw = match.group(1).replace(",", ".")
            try:
                value = float(raw)
            except ValueError:
                continue
            if value > 0:
                amounts.append(value)
    return max(amounts) if amounts else None


def _detect_prompt_injection(message: str) -> tuple[bool, Optional[float]]:
    """Detect prompt injection or abusive override requests and extract requested value."""
    normalized = _normalize(message)
    settings = get_settings()
    extracted_amount = _extract_monetary_amount(normalized)

    matched_pattern = any(re.search(pat, normalized) for pat in _INJECTION_PATTERNS)
    abusive_amount_demand = (
        extracted_amount is not None
        and (
            ("cupom" in normalized and extracted_amount > settings.max_auto_coupon_brl)
            or (
                ("reembolso" in normalized or "estorno" in normalized)
                and extracted_amount > settings.max_auto_refund_brl
                and ("ignore" in normalized or "agora" in normalized or "exijo" in normalized)
            )
        )
    )
    return (matched_pattern or abusive_amount_demand), extracted_amount


def _classify_incident_category(
    order_result: dict[str, Any],
    message: str,
    injection_detected: bool,
) -> IncidentCategory:
    """Classify the incident category using real order state plus customer context."""
    if injection_detected or not order_result.get("found"):
        return "FRAUD_RISK"

    order = order_result["order"]
    flags = order_result.get("analysis_flags", {})
    if flags.get("requires_manual_review"):
        return "FRAUD_RISK"

    normalized_msg = _normalize(message)
    status = order.get("status", "")
    delay_minutes = order.get("telemetry", {}).get("delay_minutes", 0)

    if status == "CANCELLED_BY_RESTAURANT" or "cancel" in normalized_msg:
        return "CANCELLED_ORDER"

    if status == "DELIVERED" and any(
        kw in normalized_msg
        for kw in ("falt", "nao veio", "incompleto", "refrigerante", "bebida", "item")
    ):
        return "MISSING_ITEM"

    if delay_minutes > 30 or (
        status == "IN_DELIVERY"
        and delay_minutes > 30
        and any(kw in normalized_msg for kw in ("atras", "parado", "demor"))
    ):
        return "DELAY"

    return "GENERAL_STATUS"


_ITEM_MATCH_STOPWORDS = frozenset(
    {
        "de",
        "da",
        "do",
        "das",
        "dos",
        "com",
        "sem",
        "para",
        "por",
        "meu",
        "pedido",
        "veio",
        "chegou",
        "faltou",
    }
)


def _extract_product_tokens(text: str) -> set[str]:
    """Extract meaningful product tokens excluding short words and stopwords."""
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", _normalize(text))
        if len(tok) >= 3 and tok not in _ITEM_MATCH_STOPWORDS
    }


def _match_missing_item(order: dict[str, Any], message: str) -> Optional[dict[str, Any]]:
    """Find the best matching line item from an order based on customer text."""
    msg_tokens = _extract_product_tokens(message)
    best_item: Optional[dict[str, Any]] = None
    best_overlap = 0

    for item in order.get("items", []):
        item_tokens = _extract_product_tokens(item["name"])
        overlap = len(msg_tokens & item_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_item = item

    return best_item if best_overlap > 0 else None


def _has_real_api_key(api_key: Optional[str]) -> bool:
    """Check whether an API key is configured and not a placeholder value."""
    if not api_key:
        return False
    cleaned = api_key.strip()
    return bool(cleaned) and not cleaned.lower().startswith("your_")


def _build_reasoning_prompt(state: IncidentGraphState) -> str:
    """Assemble full structured context for the reasoning step."""
    user_prompt = build_user_prompt(state["order_id"], state["customer_message"])
    order_json = json.dumps(state["order_result"], ensure_ascii=False, indent=2)
    return (
        f"{user_prompt}\n\n"
        f"## Evidência Real do Pedido (`get_order_details`)\n```json\n{order_json}\n```\n\n"
        f"## Políticas Recuperadas (`search_compensation_policy`)\n{state['policy_text']}\n\n"
        "Retorne exclusivamente um objeto JSON válido com os campos:\n"
        "- `action_type`: 'ISSUE_COUPON' | 'PARTIAL_REFUND' | 'ESCALATE_HUMAN' | 'INFORM_STATUS'\n"
        "- `amount`: número decimal >= 0.0\n"
        "- `justification`: justificativa técnica interna em português\n"
        "- `policy_reference`: código da política (ex: 'POL-DELAY-001')\n"
        "- `customer_response_draft`: resposta empática para o cliente em português"
    )


def _call_gemini_decision(
    prompt: str,
    settings: Settings,
) -> tuple[LLMDecisionProposal, int, int]:
    """Request structured JSON decision from Google Gemini via google-genai."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=5000),
    )
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=LLMDecisionProposal,
            ),
        )
        proposal = LLMDecisionProposal.model_validate_json(response.text)
        usage = getattr(response, "usage_metadata", None)
        in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
        out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
        if in_tok == 0:
            in_tok = estimate_tokens(SYSTEM_PROMPT + prompt)
        if out_tok == 0:
            out_tok = estimate_tokens(response.text or "")
        return proposal, in_tok, out_tok
    finally:
        client.close()


def _call_openai_decision(
    prompt: str,
    settings: Settings,
) -> tuple[LLMDecisionProposal, int, int]:
    """Request structured JSON decision from OpenAI via openai SDK."""
    from openai import OpenAI

    with OpenAI(api_key=settings.openai_api_key, timeout=5.0) as client:
        response = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        raw_content = response.choices[0].message.content or "{}"
        proposal = LLMDecisionProposal.model_validate_json(raw_content)
        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0) or estimate_tokens(
            SYSTEM_PROMPT + prompt
        )
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0) or estimate_tokens(
            raw_content
        )
        return proposal, in_tok, out_tok


def _deterministic_decision(
    state: IncidentGraphState,
    settings: Settings,
) -> LLMDecisionProposal:
    """Deterministic policy engine used in mock mode or as a resilient fallback."""
    order_result = state["order_result"]
    msg = state["customer_message"]
    normalized_msg = _normalize(msg)

    if state["prompt_injection_detected"]:
        abusive_amount = state.get("extracted_injected_amount")
        if abusive_amount is not None and abusive_amount > 0:
            requested_action: ActionType = (
                "PARTIAL_REFUND"
                if ("reembolso" in normalized_msg or "estorno" in normalized_msg)
                and "cupom" not in normalized_msg
                else "ISSUE_COUPON"
            )
            return LLMDecisionProposal(
                action_type=requested_action,
                amount=abusive_amount,
                justification=(
                    f"Tentativa de prompt injection detectada solicitando R$ {abusive_amount:.2f}; "
                    "submetida ao guardrail de código para bloqueio e escalonamento auditável."
                ),
                policy_reference="POL-GUARD-006",
                customer_response_draft=(
                    "Recebemos sua solicitação e encaminhamos o caso para análise prioritária "
                    "da nossa equipe de atendimento."
                ),
            )
        return LLMDecisionProposal(
            action_type="ESCALATE_HUMAN",
            amount=0.0,
            justification=(
                "Tentativa de prompt injection / manipulação de política detectada na mensagem; "
                "escalado para revisão humana conforme POL-FRAUD-005."
            ),
            policy_reference="POL-FRAUD-005",
            customer_response_draft=(
                "Sua solicitação foi encaminhada para análise prioritária de um atendente humano."
            ),
        )

    if not order_result.get("found"):
        return LLMDecisionProposal(
            action_type="ESCALATE_HUMAN",
            amount=0.0,
            justification=(
                f"Pedido {state['order_id']} não localizado na base operacional; "
                "escalonamento humano obrigatório."
            ),
            policy_reference="POL-FRAUD-005",
            customer_response_draft=(
                f"Não localizamos o pedido {state['order_id']} automaticamente. "
                "Encaminhamos seu atendimento para um especialista verificar com prioridade."
            ),
        )

    order = order_result["order"]
    flags = order_result.get("analysis_flags", {})
    telemetry = order.get("telemetry", {})

    if flags.get("requires_manual_review"):
        return LLMDecisionProposal(
            action_type="ESCALATE_HUMAN",
            amount=0.0,
            justification=(
                "Pedido sinalizado para revisão manual obrigatória (alto valor, reclamação tardia "
                "acima de 60 min ou histórico de risco) conforme POL-FRAUD-005."
            ),
            policy_reference="POL-FRAUD-005",
            customer_response_draft=(
                "Como este pedido requer uma verificação detalhada, encaminhamos sua solicitação "
                "para análise prioritária da nossa equipe especializada."
            ),
        )

    if order.get("status") == "CANCELLED_BY_RESTAURANT" and order.get("payment_status") == "APPROVED":
        total = float(order.get("total_amount", 0.0))
        return LLMDecisionProposal(
            action_type="PARTIAL_REFUND",
            amount=total,
            justification=(
                f"Pedido cancelado pelo restaurante com pagamento aprovado (R$ {total:.2f}) "
                "conforme POL-CANCEL-004."
            ),
            policy_reference="POL-CANCEL-004",
            customer_response_draft=(
                f"Sentimos muito pelo cancelamento do seu pedido de R$ {total:.2f}. "
                "Seu estorno está sendo tratado com prioridade."
            ),
        )

    if order.get("status") == "DELIVERED" and state["incident_category"] == "MISSING_ITEM":
        matched_item = _match_missing_item(order, msg)
        if matched_item is not None:
            refund_val = float(matched_item["unit_price"])
            return LLMDecisionProposal(
                action_type="PARTIAL_REFUND",
                amount=refund_val,
                justification=(
                    f"Item faltante confirmado ('{matched_item['name']}') no pedido entregue há "
                    f"{telemetry.get('delivered_minutes_ago')} min; reembolso parcial exato de "
                    f"R$ {refund_val:.2f} conforme POL-MISSING-003."
                ),
                policy_reference="POL-MISSING-003",
                customer_response_draft=(
                    f"Sentimos muito pela falta do item '{matched_item['name']}'. "
                    f"Já processamos o reembolso parcial de R$ {refund_val:.2f}."
                ),
            )
        return LLMDecisionProposal(
            action_type="ESCALATE_HUMAN",
            amount=0.0,
            justification=(
                "Cliente relatou item faltante não identificado na lista do pedido; "
                "escalado para conferência humana."
            ),
            policy_reference="POL-MISSING-003",
            customer_response_draft=(
                "Encaminhamos seu relato sobre o item faltante para conferência rápida "
                "com nossa equipe de suporte."
            ),
        )

    if order.get("status") == "IN_DELIVERY" and telemetry.get("delay_minutes", 0) > 30:
        coupon_amount = min(15.0, settings.max_auto_coupon_brl)
        return LLMDecisionProposal(
            action_type="ISSUE_COUPON",
            amount=coupon_amount,
            justification=(
                f"Atraso confirmado de {telemetry.get('delay_minutes')} min (> 30 min), "
                f"entregador parado há {telemetry.get('stopped_duration_minutes')} min "
                f"(clima {telemetry.get('weather_condition')}); cupom automático de "
                f"R$ {coupon_amount:.2f} emitido conforme POL-DELAY-001."
            ),
            policy_reference="POL-DELAY-001",
            customer_response_draft=(
                f"Pedimos desculpas pelo atraso de {telemetry.get('delay_minutes')} minutos. "
                f"Emitimos um cupom de R$ {coupon_amount:.2f} para você."
            ),
        )

    eta = telemetry.get("eta_remaining_minutes")
    return LLMDecisionProposal(
        action_type="INFORM_STATUS",
        amount=0.0,
        justification=(
            f"Pedido em rota normal dentro do prazo estimado (ETA: {eta} min, atraso: "
            f"{telemetry.get('delay_minutes', 0)} min); sem compensação financeira conforme POL-STATUS-002."
        ),
        policy_reference="POL-STATUS-002",
        customer_response_draft=(
            f"Seu pedido está a caminho dentro do prazo previsto, com entrega estimada em "
            f"cerca de {eta} minutos."
        ),
    )


def _node_inspect_order(state: IncidentGraphState) -> dict[str, Any]:
    """Step 1: Query real order telemetry and classify incident."""
    t0 = time.perf_counter()
    tool_input = {"order_id": state["order_id"]}
    order_result = get_order_details.invoke(tool_input)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    injection_detected, injected_amount = _detect_prompt_injection(state["customer_message"])
    category = _classify_incident_category(
        order_result=order_result,
        message=state["customer_message"],
        injection_detected=injection_detected,
    )

    tool_log = ToolCallLog(
        step=len(state["tools_executed"]) + 1,
        tool_name="get_order_details",
        tool_input=tool_input,
        tool_output=order_result,
        latency_ms=round(latency_ms, 3),
        guardrail_triggered=False,
    ).model_dump()

    return {
        "order_result": order_result,
        "incident_category": category,
        "prompt_injection_detected": injection_detected,
        "extracted_injected_amount": injected_amount,
        "tools_executed": [*state["tools_executed"], tool_log],
        "state_transitions": [*state["state_transitions"], "inspect_order"],
    }


def _node_retrieve_policy(state: IncidentGraphState) -> dict[str, Any]:
    """Step 2: Retrieve relevant compensation policies via lightweight RAG."""
    t0 = time.perf_counter()
    tool_input = {
        "incident_category": state["incident_category"],
        "query": state["customer_message"],
    }
    policy_text = search_compensation_policy.invoke(tool_input)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    tool_log = ToolCallLog(
        step=len(state["tools_executed"]) + 1,
        tool_name="search_compensation_policy",
        tool_input=tool_input,
        tool_output=policy_text,
        latency_ms=round(latency_ms, 3),
        guardrail_triggered=False,
    ).model_dump()

    return {
        "policy_text": policy_text,
        "tools_executed": [*state["tools_executed"], tool_log],
        "state_transitions": [*state["state_transitions"], "retrieve_policy"],
    }


def _node_reason_decision(state: IncidentGraphState) -> dict[str, Any]:
    """Step 3: Select action via configured LLM provider with resilient fallback."""
    settings = get_settings()
    prompt = _build_reasoning_prompt(state)
    provider_used = "mock"
    model_used = "mock-deterministic-v1"
    in_tok = estimate_tokens(SYSTEM_PROMPT + prompt)
    out_tok = 0
    proposal: Optional[LLMDecisionProposal] = None

    if settings.llm_provider == "gemini" and _has_real_api_key(settings.gemini_api_key):
        try:
            proposal, in_tok, out_tok = _call_gemini_decision(prompt, settings)
            provider_used = "gemini"
            model_used = settings.gemini_model
        except Exception as exc:
            _LOGGER.warning(
                "Gemini call failed; falling back to deterministic engine.",
                extra={"trace_id": state["trace_id"], "error": str(exc)},
            )
    elif settings.llm_provider == "openai" and _has_real_api_key(settings.openai_api_key):
        try:
            proposal, in_tok, out_tok = _call_openai_decision(prompt, settings)
            provider_used = "openai"
            model_used = settings.openai_model
        except Exception as exc:
            _LOGGER.warning(
                "OpenAI call failed; falling back to deterministic engine.",
                extra={"trace_id": state["trace_id"], "error": str(exc)},
            )

    if proposal is None:
        proposal = _deterministic_decision(state, settings)
        out_tok = estimate_tokens(proposal.model_dump_json())

    return {
        "proposed_action": proposal.action_type,
        "proposed_amount": proposal.amount,
        "proposed_justification": proposal.justification,
        "policy_reference": proposal.policy_reference,
        "customer_response_draft": proposal.customer_response_draft,
        "input_tokens": state["input_tokens"] + in_tok,
        "output_tokens": state["output_tokens"] + out_tok,
        "provider_used": provider_used,
        "model_used": model_used,
        "state_transitions": [*state["state_transitions"], "reason_decision"],
    }


def _node_execute_resolution(state: IncidentGraphState) -> dict[str, Any]:
    """Step 4: Execute final decision through apply_resolution with hard code guardrails."""
    t0 = time.perf_counter()
    tool_input = {
        "order_id": state["order_id"],
        "action_type": state["proposed_action"],
        "amount": state["proposed_amount"],
        "justification": state["proposed_justification"],
    }
    resolution_output = apply_resolution.invoke(tool_input)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    guardrail_triggered = bool(resolution_output.get("guardrail_triggered", False))
    tool_log = ToolCallLog(
        step=len(state["tools_executed"]) + 1,
        tool_name="apply_resolution",
        tool_input=tool_input,
        tool_output=resolution_output,
        latency_ms=round(latency_ms, 3),
        guardrail_triggered=guardrail_triggered,
    ).model_dump()

    decision = DecisionOutput(
        action_type=resolution_output["action_type"],
        amount=float(resolution_output["amount"]),
        justification=resolution_output["justification"],
        policy_reference=state.get("policy_reference"),
        guardrail_overridden=guardrail_triggered,
    ).model_dump()

    return {
        "resolution_output": resolution_output,
        "decision": decision,
        "tools_executed": [*state["tools_executed"], tool_log],
        "state_transitions": [*state["state_transitions"], "execute_resolution"],
    }


def _node_craft_response(state: IncidentGraphState) -> dict[str, Any]:
    """Step 5: Build clear, empathetic customer message aligned with effective decision."""
    decision = state["decision"]
    action = decision["action_type"]
    amount = float(decision["amount"])
    overridden = bool(decision.get("guardrail_overridden", False))
    order_result = state["order_result"]
    order = order_result.get("order") or {}
    customer_name = order.get("customer", {}).get("name")
    greeting = f"Olá, {customer_name.split()[0]}!" if customer_name else "Olá!"
    telemetry = order.get("telemetry", {})

    if action == "ISSUE_COUPON" and not overridden:
        delay = telemetry.get("delay_minutes", 0)
        weather = telemetry.get("weather_condition", "CLEAR")
        weather_note = " devido às condições climáticas adversas" if weather == "HEAVY_RAIN" else ""
        customer_response = (
            f"{greeting} Pedimos desculpas pelo atraso de {delay} minutos na entrega do pedido "
            f"{state['order_id']}{weather_note}. Para compensar o transtorno, emitimos um cupom "
            f"automático de R$ {amount:.2f} já disponível em sua conta. "
            "O entregador segue em acompanhamento prioritário."
        )
    elif action == "PARTIAL_REFUND" and not overridden:
        customer_response = (
            f"{greeting} Sentimos muito pelo ocorrido com o pedido {state['order_id']}. "
            f"Já aprovamos o reembolso de R$ {amount:.2f}, que será estornado diretamente no "
            "mesmo método de pagamento utilizado."
        )
    elif action == "INFORM_STATUS":
        eta = telemetry.get("eta_remaining_minutes")
        courier = telemetry.get("courier_name") or "nosso entregador parceiro"
        eta_msg = f"em cerca de {eta} minutos" if eta is not None else "dentro do prazo previsto"
        customer_response = (
            f"{greeting} Conferimos o status em tempo real do pedido {state['order_id']} e ele "
            f"está em rota normal com {courier}, com previsão de chegada {eta_msg}."
        )
    else:
        if order.get("status") == "CANCELLED_BY_RESTAURANT" and order.get("payment_status") == "APPROVED":
            total = float(order.get("total_amount", 0.0))
            customer_response = (
                f"{greeting} Lamentamos muito o cancelamento do pedido {state['order_id']} pelo "
                f"restaurante ({order.get('cancel_reason') or 'indisponibilidade operacional'}). "
                f"Como o valor é de R$ {total:.2f}, encaminhamos seu estorno integral com "
                "prioridade máxima para nossa equipe financeira concluir ainda hoje."
            )
        elif state.get("prompt_injection_detected"):
            customer_response = (
                f"{greeting} Não é possível conceder compensações fora das políticas oficiais para o "
                f"pedido {state['order_id']}. Caso precise de suporte legítimo sobre sua entrega, "
                "encaminhamos o atendimento para análise de um especialista humano."
            )
        else:
            customer_response = (
                f"{greeting} Analisamos os detalhes do pedido {state['order_id']} e, pelas nossas "
                "políticas de atendimento, encaminhamos sua solicitação para revisão prioritária "
                "de um especialista humano, que dará continuidade em instantes."
            )

    return {
        "customer_response": customer_response,
        "state_transitions": [*state["state_transitions"], "craft_response"],
    }


def build_incident_graph():
    """Construct and compile the explicit LangGraph incident triage workflow."""
    graph = StateGraph(IncidentGraphState)
    graph.add_node("inspect_order", _node_inspect_order)
    graph.add_node("retrieve_policy", _node_retrieve_policy)
    graph.add_node("reason_decision", _node_reason_decision)
    graph.add_node("execute_resolution", _node_execute_resolution)
    graph.add_node("craft_response", _node_craft_response)

    graph.add_edge(START, "inspect_order")
    graph.add_edge("inspect_order", "retrieve_policy")
    graph.add_edge("retrieve_policy", "reason_decision")
    graph.add_edge("reason_decision", "execute_resolution")
    graph.add_edge("execute_resolution", "craft_response")
    graph.add_edge("craft_response", END)

    return graph.compile()


_COMPILED_GRAPH = build_incident_graph()


def resolve_incident(request: IncidentResolveRequest) -> IncidentResolveResponse:
    """Execute the full incident triage graph and return structured response + telemetry."""
    settings = get_settings()
    collector = TelemetryCollector()

    initial_state: IncidentGraphState = {
        "trace_id": collector.trace_id,
        "order_id": request.order_id.strip().upper(),
        "customer_message": request.customer_message.strip(),
        "incident_category": "GENERAL_STATUS",
        "prompt_injection_detected": False,
        "extracted_injected_amount": None,
        "order_result": {},
        "policy_text": "",
        "proposed_action": "INFORM_STATUS",
        "proposed_amount": 0.0,
        "proposed_justification": "",
        "policy_reference": "POL-STATUS-002",
        "customer_response_draft": "",
        "resolution_output": {},
        "decision": {},
        "customer_response": "",
        "tools_executed": [],
        "state_transitions": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "provider_used": settings.llm_provider,
        "model_used": settings.active_model,
    }

    final_state = _COMPILED_GRAPH.invoke(initial_state)

    collector.add_tokens(
        input_tokens=final_state["input_tokens"],
        output_tokens=final_state["output_tokens"],
    )
    collector.tools_executed = [
        ToolCallLog.model_validate(item) for item in final_state["tools_executed"]
    ]
    collector.state_transitions = list(final_state["state_transitions"])

    telemetry = collector.build_output(
        provider=final_state["provider_used"],
        model=final_state["model_used"],
        settings=settings,
    )
    decision = DecisionOutput.model_validate(final_state["decision"])

    _LOGGER.info(
        "Incident triage completed",
        extra={
            "trace_id": telemetry.trace_id,
            "order_id": final_state["order_id"],
            "action_type": decision.action_type,
            "amount": decision.amount,
            "guardrail_overridden": decision.guardrail_overridden,
            "latency_ms": telemetry.latency_ms,
            "provider": telemetry.provider,
            "model": telemetry.model,
            "estimated_cost_usd": telemetry.estimated_cost_usd,
        },
    )

    return IncidentResolveResponse(
        order_id=final_state["order_id"],
        customer_response=final_state["customer_response"],
        decision=decision,
        telemetry=telemetry,
    )
