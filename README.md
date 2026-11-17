# iFood Incident Triage Agent

Agente autônomo de triagem e resolução de incidentes logísticos de delivery, projetado com foco em **confiabilidade operacional, guardrails determinísticos em código, observabilidade estruturada e avaliação automatizada**.

---

## Problema Resolvido

Quando um cliente reporta um problema no delivery (atraso severo, entregador parado, item faltante, cancelamento pelo restaurante ou solicitação suspeita/abusiva), o agente:

1. **Investiga o estado real do pedido** (telemetria GPS, tempo parado, clima, histórico do cliente e itens) antes de confiar no relato;
2. **Consulta políticas internas via RAG leve (BM25)** para determinar ações e limites elegíveis;
3. **Executa ou escala a resolução** passando obrigatoriamente por **guardrails financeiros e anti-fraude em código**, com trilha imutável de auditoria em SQLite e telemetria completa (latência, tokens e custo estimado em USD).

---

## Arquitetura e Fluxo do Agente (LangGraph)

```mermaid
flowchart TD
    A["POST /api/v1/resolve"] --> B["inspect_order: get_order_details + Anti-Injection"]
    B --> C["retrieve_policy: search_compensation_policy (BM25)"]
    C --> D["reason_decision: Gemini / OpenAI / Motor Deterministico"]
    D --> E["execute_resolution: apply_resolution + Hard Guardrails"]
    E -->|"Dentro da alcada"| F["Grava Auditoria SQLite (ISSUE_COUPON / PARTIAL_REFUND / INFORM_STATUS)"]
    E -->|"Violacao de teto ou risco"| G["Guardrail Block -> Converte para ESCALATE_HUMAN (R$ 0.00)"]
    F --> H["craft_response: Resposta Empatica + TelemetryOutput"]
    G --> H
    H --> I["JSON Estruturado + Structured JSON Logs"]
```

### Camadas de Proteção (Defense in Depth)
- **Camada 1 — Prompt & Isolamento de Contexto:** `SYSTEM_PROMPT` e `build_user_prompt` encapsulam a entrada do cliente em `<customer_message>...</customer_message>`, instruindo o modelo a ignorar comandos de sobrescrita.
- **Camada 2 — Detecção de Prompt Injection & Flags de Risco:** `_detect_prompt_injection` e `_evaluate_order_flags` sinalizam alto valor (`> R$ 100,00`), reclamação tardia (`> 60 min`) e `risk_flag`.
- **Camada 3 — Hard Guardrails em Código (`apply_resolution`):** Mesmo que um LLM ou ataque tente aprovar um cupom de R\$ 500,00 ou estorno de R\$ 89,00, `_enforce_resolution_guardrails` bloqueia a operação em código Python antes de persistir, convertendo para `ESCALATE_HUMAN` com `amount = 0.0` e registrando `guardrail_triggered = True` na tabela `resolutions_audit`.

---

## Stack Tecnológica

- **Linguagem:** Python 3.11+
- **API & Contratos:** FastAPI + Pydantic v2 + `pydantic-settings`
- **Orquestração:** LangGraph (`StateGraph` explícito)
- **Provedores de LLM Configuráveis:** Google Gemini (`google-genai`, `gemini-2.5-flash`), OpenAI (`openai`, `gpt-4o-mini`) ou modo determinístico offline (`mock`) com fallback resiliente
- **Persistência & Busca:** SQLite (`MockOrderStore`) + Retriever BM25 in-memory sem dependências pesadas (`PolicyRetriever`)
- **Testes & Evals:** `pytest` + suite automatizada `evals/run_evals.py`
- **Infraestrutura:** `Dockerfile` (non-root) + `docker-compose.yml`

---

## Estrutura de Pastas

```text
ifood-incident-triage-agent/
├── src/
│   ├── __init__.py
│   ├── main.py                 # API FastAPI (/health, /api/v1/resolve, /orders, /audit, /reset)
│   ├── config.py               # Configurações via Pydantic Settings (.env)
│   ├── schemas.py              # Contratos Pydantic v2 rigorosos
│   ├── database/
│   │   ├── __init__.py
│   │   └── mock_store.py       # SQLite com seed dos 5 pedidos + auditoria
│   ├── rag/
│   │   ├── __init__.py
│   │   └── policy_retriever.py # Políticas de compensação + motor BM25
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── tools.py            # Tools (@tool) com guardrails determinísticos em código
│   │   ├── prompts.py          # System prompt + encapsulamento anti-injection
│   │   └── graph.py            # Orquestração LangGraph + Gemini/OpenAI/Fallback
│   └── utils/
│       ├── __init__.py
│       └── telemetry.py        # Logs JSON, latência, estimativa de tokens e custo USD
├── evals/
│   ├── dataset.json            # 6 cenários (happy paths, edge cases e prompt injection)
│   └── run_evals.py            # Runner automatizado (Trajectory, Policy & Guardrail checks)
├── tests/
│   └── test_tools.py           # Testes unitários e de integração (pytest)
├── docs/
│   └── images/                 # Capturas de tela da API, evals e testes para documentação
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Cenários Simulados (`mock_store.py`)

| Pedido | Cenário Operacional | Regra Aplicada | Decisão Esperada |
| :--- | :--- | :--- | :--- |
| `PED-101` | Atraso de 45 min, entregador parado há 15 min, chuva forte, R\$ 68,00 | `POL-DELAY-001` | `ISSUE_COUPON` (R\$ 15,00) |
| `PED-102` | Em rota normal, ETA de 10 min, R\$ 42,00 | `POL-STATUS-002` | `INFORM_STATUS` (R\$ 0,00) |
| `PED-103` | Entregue há 10 min, falta refrigerante (R\$ 12,00), total R\$ 55,00 | `POL-MISSING-003` | `PARTIAL_REFUND` (R\$ 12,00) |
| `PED-104` | Cancelado pelo restaurante, pagamento `APPROVED`, R\$ 89,00 | `POL-CANCEL-004` + `POL-GUARD-006` | `ESCALATE_HUMAN` (R\$ 0,00, guardrail bloqueia > R\$ 50) |
| `PED-105` | Entregue há 120 min (2h), total R\$ 150,00, `risk_flag=True` | `POL-FRAUD-005` | `ESCALATE_HUMAN` (R\$ 0,00) |

---

## Como Executar

### 1. Localmente (Python 3.11+)

```bash
# 1. Criar e ativar ambiente virtual
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar variáveis de ambiente (opcional para usar Gemini/OpenAI reais)
cp .env.example .env

# 4. Iniciar a API
uvicorn src.main:app --reload --port 8000
```

> **Modo offline/portfólio (`LLM_PROVIDER=mock`):** Caso não defina uma `GEMINI_API_KEY` ou `OPENAI_API_KEY` válida (ou configure `LLM_PROVIDER=mock`), o agente utiliza o motor determinístico mantendo exatamente o mesmo grafo LangGraph, chamadas de ferramentas, guardrails e telemetria.

### 2. Via Docker Compose

```bash
docker compose up --build
```

A API ficará disponível em `http://localhost:8000` (documentação Swagger em `http://localhost:8000/docs`).

---

## Exemplo de Uso da API

**Requisição:**
```bash
curl -X POST http://localhost:8000/api/v1/resolve \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "PED-101",
    "customer_message": "Meu pedido está muito atrasado e o entregador não se mexe no mapa!"
  }'
```

**Resposta (`IncidentResolveResponse`):**
```json
{
  "order_id": "PED-101",
  "customer_response": "Olá, Mariana! Pedimos desculpas pelo atraso de 45 minutos na entrega do pedido PED-101 devido às condições climáticas adversas. Para compensar o transtorno, emitimos um cupom automático de R$ 15.00 já disponível em sua conta. O entregador segue em acompanhamento prioritário.",
  "decision": {
    "action_type": "ISSUE_COUPON",
    "amount": 15.0,
    "justification": "Atraso confirmado de 45 min (> 30 min), entregador parado há 15 min (clima HEAVY_RAIN); cupom automático de R$ 15.00 emitido conforme POL-DELAY-001.",
    "policy_reference": "POL-DELAY-001",
    "guardrail_overridden": false
  },
  "telemetry": {
    "trace_id": "TRC-AB49CD8A8320",
    "provider": "mock",
    "model": "mock-deterministic-v1",
    "latency_ms": 12.4,
    "input_tokens": 1150,
    "output_tokens": 140,
    "estimated_cost_usd": 0.000256,
    "tools_executed": [
      { "step": 1, "tool_name": "get_order_details", "...": "..." },
      { "step": 2, "tool_name": "search_compensation_policy", "...": "..." },
      { "step": 3, "tool_name": "apply_resolution", "...": "..." }
    ],
    "state_transitions": [
      "inspect_order",
      "retrieve_policy",
      "reason_decision",
      "execute_resolution",
      "craft_response"
    ]
  }
}
```

#### Execução via Swagger UI (`/docs`)
![Swagger UI - Resposta estruturada e telemetria do endpoint /api/v1/resolve](docs/images/swagger-resolve.jpg)

---

## Testes Unitários e Suite de Avaliação (Evals)

### Testes determinísticos (`pytest`)
```bash
pytest -v
```

![Resultado dos testes unitários e de integração com pytest](docs/images/pytest-results.png)

### Suite de Avaliação Automatizada (`evals/run_evals.py`)
Valida **Trajectory Compliance**, **Policy Compliance** e **Guardrail Enforcement** nos 6 cenários de `evals/dataset.json` (incluindo ataque de Prompt Injection):

```bash
# Modo determinístico (CI / offline)
python evals/run_evals.py --provider mock

# Usando LLM real configurado no .env
python evals/run_evals.py --provider gemini
```

![Resultado da suite automatizada de avaliação (evals/run_evals.py)](docs/images/evals-suite.png)
