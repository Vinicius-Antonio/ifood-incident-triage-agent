"""Pydantic v2 domain, API, and observability contracts."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

ActionType = Literal[
    "ISSUE_COUPON",
    "PARTIAL_REFUND",
    "ESCALATE_HUMAN",
    "INFORM_STATUS",
]

IncidentCategory = Literal[
    "DELAY",
    "MISSING_ITEM",
    "CANCELLED_ORDER",
    "FRAUD_RISK",
    "GENERAL_STATUS",
]


class OrderItem(BaseModel):
    """Single line item belonging to an order."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    name: str
    quantity: int = Field(..., ge=1)
    unit_price: float = Field(..., ge=0.0)

    @property
    def subtotal(self) -> float:
        """Compute total line item price."""
        return round(self.quantity * self.unit_price, 2)


class DeliveryTelemetry(BaseModel):
    """Real-time logistics and courier telemetry for an order."""

    model_config = ConfigDict(extra="forbid")

    courier_name: Optional[str] = None
    courier_status: str
    stopped_duration_minutes: int = Field(default=0, ge=0)
    weather_condition: str = "CLEAR"
    eta_remaining_minutes: Optional[int] = Field(default=None, ge=0)
    delay_minutes: int = Field(default=0, ge=0)
    delivered_minutes_ago: Optional[int] = Field(default=None, ge=0)
    gps_coordinates: Optional[str] = None


class CustomerProfile(BaseModel):
    """Customer profile and historical risk indicators."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str
    name: str
    tier: Literal["STANDARD", "CLUB", "VIP"] = "STANDARD"
    total_orders: int = Field(default=0, ge=0)
    prior_refund_requests_30d: int = Field(default=0, ge=0)
    is_recurrent: bool = False
    risk_flag: bool = False


class OrderRecord(BaseModel):
    """Full order snapshot stored in the operational database."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    restaurant_name: str
    status: str
    total_amount: float = Field(..., ge=0.0)
    payment_status: str
    cancel_reason: Optional[str] = None
    items: list[OrderItem] = Field(default_factory=list)
    telemetry: DeliveryTelemetry
    customer: CustomerProfile
    created_at: str


class IncidentResolveRequest(BaseModel):
    """Incoming payload for POST /api/v1/resolve."""

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(..., min_length=1, description="Identificador único do pedido (ex: PED-101)")
    customer_message: str = Field(
        ...,
        min_length=1,
        description="Mensagem do cliente descrevendo o incidente logístico",
    )

    @field_validator("order_id", "customer_message")
    @classmethod
    def _reject_blank_strings(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or whitespace only.")
        return cleaned


class ToolCallLog(BaseModel):
    """Execution trace for a single agent tool invocation."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(..., ge=1)
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_output: Any = None
    latency_ms: float = Field(..., ge=0.0)
    guardrail_triggered: bool = False


class DecisionOutput(BaseModel):
    """Structured final operational decision emitted by the agent."""

    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    amount: float = Field(default=0.0, ge=0.0)
    justification: str = Field(..., min_length=1)
    policy_reference: Optional[str] = None
    guardrail_overridden: bool = False


class TelemetryOutput(BaseModel):
    """Execution metrics, cost estimate, and ordered trace of tool calls."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    provider: str
    model: str
    latency_ms: float = Field(..., ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    tools_executed: list[ToolCallLog] = Field(default_factory=list)
    state_transitions: list[str] = Field(default_factory=list)


class AuditTrailEntry(BaseModel):
    """Immutable audit record persisted for every applied resolution."""

    model_config = ConfigDict(extra="forbid")

    audit_id: str
    timestamp: str
    order_id: str
    action_type: ActionType
    amount: float = Field(..., ge=0.0)
    justification: str
    guardrail_triggered: bool = False
    raw_requested_action: Optional[str] = None
    raw_requested_amount: Optional[float] = None


class IncidentResolveResponse(BaseModel):
    """Structured API response returned by POST /api/v1/resolve."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    customer_response: str
    decision: DecisionOutput
    telemetry: TelemetryOutput
