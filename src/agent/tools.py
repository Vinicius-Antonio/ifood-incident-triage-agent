"""Typed agent tools with deterministic financial and operational guardrails."""

import math
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from langchain_core.tools import tool

from src.config import Settings, get_settings
from src.database.mock_store import get_order_by_id, save_audit_entry
from src.rag.policy_retriever import retrieve_policies
from src.schemas import ActionType, AuditTrailEntry, OrderRecord

_VALID_ACTIONS: tuple[ActionType, ...] = (
    "ISSUE_COUPON",
    "PARTIAL_REFUND",
    "ESCALATE_HUMAN",
    "INFORM_STATUS",
)


def _evaluate_order_flags(order: OrderRecord, settings: Settings) -> dict[str, bool]:
    """Compute operational risk and eligibility flags from real order telemetry."""
    is_high_value = order.total_amount > settings.high_value_order_threshold_brl
    is_late_claim = (
        order.telemetry.delivered_minutes_ago is not None
        and order.telemetry.delivered_minutes_ago
        > settings.max_delivered_minutes_for_auto_refund
    )
    eligible_for_delay_coupon = (
        order.status == "IN_DELIVERY"
        and order.telemetry.delay_minutes > 30
        and not order.customer.risk_flag
        and not is_high_value
    )
    requires_manual_review = (
        is_high_value
        or is_late_claim
        or order.customer.risk_flag
        or order.customer.prior_refund_requests_30d >= 3
    )
    return {
        "is_high_value": is_high_value,
        "is_late_claim": is_late_claim,
        "eligible_for_delay_coupon": eligible_for_delay_coupon,
        "requires_manual_review": requires_manual_review,
    }


def _enforce_resolution_guardrails(
    order: Optional[OrderRecord],
    action_type: str,
    amount: float,
    settings: Settings,
) -> tuple[ActionType, float, bool, list[str]]:
    """Apply deterministic code-level guardrails over any LLM resolution proposal."""
    reasons: list[str] = []
    guardrail_triggered = False

    if action_type not in _VALID_ACTIONS:
        return (
            "ESCALATE_HUMAN",
            0.0,
            True,
            [f"Ação '{action_type}' inválida bloqueada pelo guardrail."],
        )

    effective_action: ActionType = action_type  # type: ignore[assignment]
    try:
        numeric_amount = float(amount)
    except (TypeError, ValueError):
        return (
            "ESCALATE_HUMAN",
            0.0,
            True,
            ["Valor financeiro não numérico bloqueado pelo guardrail."],
        )

    if math.isnan(numeric_amount) or math.isinf(numeric_amount) or numeric_amount < 0:
        return (
            "ESCALATE_HUMAN",
            0.0,
            True,
            [f"Valor financeiro inválido ({amount}) bloqueado pelo guardrail."],
        )

    effective_amount = round(numeric_amount, 2)

    if effective_action == "ISSUE_COUPON" and effective_amount > settings.max_auto_coupon_brl:
        guardrail_triggered = True
        reasons.append(
            f"Cupom solicitado (R$ {effective_amount:.2f}) excede teto automático "
            f"de R$ {settings.max_auto_coupon_brl:.2f}."
        )

    if (
        effective_action == "PARTIAL_REFUND"
        and effective_amount > settings.max_auto_refund_brl
    ):
        guardrail_triggered = True
        reasons.append(
            f"Reembolso solicitado (R$ {effective_amount:.2f}) excede teto automático "
            f"de R$ {settings.max_auto_refund_brl:.2f}."
        )

    if effective_action in ("ISSUE_COUPON", "PARTIAL_REFUND"):
        if order is None:
            guardrail_triggered = True
            reasons.append("Compensação financeira bloqueada para pedido inexistente.")
        else:
            if order.total_amount > settings.high_value_order_threshold_brl:
                guardrail_triggered = True
                reasons.append(
                    f"Pedido de alto valor (R$ {order.total_amount:.2f} > "
                    f"R$ {settings.high_value_order_threshold_brl:.2f}) exige revisão humana."
                )
            if (
                order.telemetry.delivered_minutes_ago is not None
                and order.telemetry.delivered_minutes_ago
                > settings.max_delivered_minutes_for_auto_refund
            ):
                guardrail_triggered = True
                reasons.append(
                    f"Pedido entregue há {order.telemetry.delivered_minutes_ago} min "
                    f"(> {settings.max_delivered_minutes_for_auto_refund} min) exige revisão humana."
                )
            if order.customer.risk_flag:
                guardrail_triggered = True
                reasons.append("Cliente sinalizado com risk_flag=True exige revisão humana.")

    if guardrail_triggered:
        return "ESCALATE_HUMAN", 0.0, True, reasons

    if effective_action in ("ESCALATE_HUMAN", "INFORM_STATUS") and effective_amount > 0.0:
        return (
            effective_action,
            0.0,
            True,
            [
                f"Ação '{effective_action}' não permite compensação financeira "
                f"(valor R$ {effective_amount:.2f} zerado pelo guardrail)."
            ],
        )

    return effective_action, effective_amount, False, reasons


@tool
def get_order_details(order_id: str) -> dict[str, Any]:
    """Consulta os dados completos do pedido, itens, perfil do cliente e telemetria logística.

    Use esta ferramenta sempre como primeiro passo para verificar o estado real do pedido.
    """
    try:
        cleaned_id = (order_id or "").strip()
        if not cleaned_id:
            return {
                "found": False,
                "order_id": order_id,
                "error": "O identificador order_id não pode ser vazio.",
            }

        order = get_order_by_id(cleaned_id)
        if order is None:
            return {
                "found": False,
                "order_id": cleaned_id,
                "error": f"Pedido '{cleaned_id}' não encontrado na base operacional.",
            }

        flags = _evaluate_order_flags(order, get_settings())
        return {
            "found": True,
            "order": order.model_dump(),
            "analysis_flags": flags,
        }
    except Exception as exc:
        return {
            "found": False,
            "order_id": str(order_id),
            "error": f"Erro ao consultar pedido: {exc}",
        }


@tool
def search_compensation_policy(incident_category: str, query: str) -> str:
    """Consulta o manual interno de políticas de compensação, cupom, estorno e escalonamento.

    Args:
        incident_category: Categoria do incidente (DELAY, MISSING_ITEM, CANCELLED_ORDER,
            FRAUD_RISK ou GENERAL_STATUS).
        query: Descrição curta do caso para busca semântica nas regras internas.
    """
    try:
        settings = get_settings()
        policies = retrieve_policies(
            incident_category=incident_category,
            query=query,
            top_k=3,
            include_guardrails=True,
        )

        lines = [
            "=== LIMITES OPERACIONAIS VIGENTES ===",
            f"- Teto cupom automático (ISSUE_COUPON): R$ {settings.max_auto_coupon_brl:.2f}",
            f"- Teto reembolso automático (PARTIAL_REFUND): R$ {settings.max_auto_refund_brl:.2f}",
            f"- Limite pedido alto valor (escalonamento obrigatório): > R$ {settings.high_value_order_threshold_brl:.2f}",
            f"- Limite janela pós-entrega para auto-reembolso: <= {settings.max_delivered_minutes_for_auto_refund} min",
            "",
            "=== POLÍTICAS RECUPERADAS ===",
        ]
        for policy in policies:
            lines.append(
                f"[{policy['policy_id']}] ({policy['category']}) {policy['title']} "
                f"| Ações permitidas: {', '.join(policy['allowed_actions'])} "
                f"| Teto: R$ {policy['max_auto_amount_brl']:.2f}\n"
                f"Regra: {policy['content']}"
            )
        return "\n\n".join(lines)
    except Exception as exc:
        return f"Erro ao consultar políticas internas: {exc}"


@tool
def apply_resolution(
    order_id: str,
    action_type: Literal["ISSUE_COUPON", "PARTIAL_REFUND", "ESCALATE_HUMAN", "INFORM_STATUS"],
    amount: float,
    justification: str,
) -> dict[str, Any]:
    """Executa e audita a decisão operacional final do incidente com guardrails financeiros rígidos.

    Args:
        order_id: Identificador do pedido (ex: PED-101).
        action_type: Ação final (ISSUE_COUPON, PARTIAL_REFUND, ESCALATE_HUMAN ou INFORM_STATUS).
        amount: Valor em reais (BRL) para cupom/reembolso; deve ser 0.0 para ESCALATE_HUMAN ou INFORM_STATUS.
        justification: Justificativa técnica interna citando evidências do pedido e da política.
    """
    try:
        settings = get_settings()
        normalized_order_id = (order_id or "").strip().upper() or "UNKNOWN"
        order = get_order_by_id(normalized_order_id) if normalized_order_id != "UNKNOWN" else None

        effective_action, effective_amount, guardrail_triggered, reasons = (
            _enforce_resolution_guardrails(
                order=order,
                action_type=action_type,
                amount=amount,
                settings=settings,
            )
        )

        clean_justification = (justification or "").strip() or (
            "Decisão registrada pelo agente de triagem."
        )
        if guardrail_triggered and reasons:
            clean_justification = f"{clean_justification} [GUARDRAIL BLOCK] {' '.join(reasons)}"

        audit_entry = AuditTrailEntry(
            audit_id=f"AUD-{uuid.uuid4().hex[:10].upper()}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            order_id=normalized_order_id,
            action_type=effective_action,
            amount=effective_amount,
            justification=clean_justification,
            guardrail_triggered=guardrail_triggered,
            raw_requested_action=str(action_type),
            raw_requested_amount=float(amount) if isinstance(amount, (int, float)) else None,
        )
        save_audit_entry(audit_entry)

        return {
            "status": "SUCCESS",
            "audit_id": audit_entry.audit_id,
            "order_id": audit_entry.order_id,
            "action_type": effective_action,
            "amount": effective_amount,
            "justification": clean_justification,
            "guardrail_triggered": guardrail_triggered,
            "raw_requested_action": audit_entry.raw_requested_action,
            "raw_requested_amount": audit_entry.raw_requested_amount,
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "order_id": str(order_id),
            "action_type": "ESCALATE_HUMAN",
            "amount": 0.0,
            "justification": f"Falha ao aplicar resolução; escalado por segurança: {exc}",
            "guardrail_triggered": True,
        }


AGENT_TOOLS = [
    get_order_details,
    search_compensation_policy,
    apply_resolution,
]
