"""FastAPI entrypoint for the iFood Incident Triage Agent service."""

from typing import Optional

from fastapi import FastAPI, Query

from src.agent.graph import resolve_incident
from src.config import get_settings
from src.database.mock_store import get_audit_entries, list_all_orders, reset_store
from src.schemas import (
    AuditTrailEntry,
    IncidentResolveRequest,
    IncidentResolveResponse,
    OrderRecord,
)

_settings = get_settings()

app = FastAPI(
    title=_settings.app_name,
    version="0.1.0",
    description=(
        "Agente Autônomo de Triagem e Resolução de Incidentes Logísticos "
        "com RAG leve, guardrails determinísticos em código e telemetria estruturada."
    ),
)


@app.get("/health")
def health_check() -> dict[str, object]:
    """Return service health, active LLM configuration, and seeded order count."""
    settings = get_settings()
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "provider": settings.llm_provider,
        "model": settings.active_model,
        "orders_seeded": len(list_all_orders()),
    }


@app.post("/api/v1/resolve", response_model=IncidentResolveResponse)
def resolve_order_incident(payload: IncidentResolveRequest) -> IncidentResolveResponse:
    """Investigate and resolve a customer delivery incident with full auditability."""
    return resolve_incident(payload)


@app.get("/api/v1/orders", response_model=list[OrderRecord])
def get_orders() -> list[OrderRecord]:
    """List all mocked orders currently stored in SQLite."""
    return list_all_orders()


@app.get("/api/v1/audit", response_model=list[AuditTrailEntry])
def get_audit_trail(
    order_id: Optional[str] = Query(default=None, description="Filtrar por ID do pedido"),
) -> list[AuditTrailEntry]:
    """Return persisted audit entries for executed resolutions."""
    return get_audit_entries(order_id=order_id)


@app.post("/api/v1/reset")
def reset_operational_store() -> dict[str, object]:
    """Reset the SQLite mock store and clear audit records."""
    reset_store()
    return {
        "status": "reset",
        "orders_seeded": len(list_all_orders()),
    }
