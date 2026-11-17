"""Structured JSON logging, token estimation, and execution telemetry."""

import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from src.config import Settings, get_settings
from src.schemas import TelemetryOutput, ToolCallLog

_RESERVED_LOG_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class JsonLogFormatter(logging.Formatter):
    """Format log records as single-line structured JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str = "ifood_triage_agent") -> logging.Logger:
    """Return a structured JSON logger without duplicating handlers."""
    logger = logging.getLogger(name)
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def estimate_tokens(text: str) -> int:
    """Estimate token count deterministically (~4 characters per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    settings: Optional[Settings] = None,
) -> float:
    """Compute estimated USD cost from token counts and configured unit rates."""
    cfg = settings or get_settings()
    input_cost = (max(0, input_tokens) / 1_000_000.0) * cfg.input_cost_per_1m_usd
    output_cost = (max(0, output_tokens) / 1_000_000.0) * cfg.output_cost_per_1m_usd
    return round(input_cost + output_cost, 8)


class TelemetryCollector:
    """Tracks per-run latency, state transitions, tool calls, tokens, and USD cost."""

    def __init__(self, trace_id: Optional[str] = None) -> None:
        self.trace_id = trace_id or f"TRC-{uuid.uuid4().hex[:12].upper()}"
        self._start_time = time.perf_counter()
        self.input_tokens = 0
        self.output_tokens = 0
        self.tools_executed: list[ToolCallLog] = []
        self.state_transitions: list[str] = []

    def add_transition(self, node_name: str) -> None:
        """Record an agent state machine transition."""
        self.state_transitions.append(node_name)

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate input and output token usage."""
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)

    def record_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
        latency_ms: float,
        guardrail_triggered: bool = False,
    ) -> ToolCallLog:
        """Append an ordered tool execution entry."""
        entry = ToolCallLog(
            step=len(self.tools_executed) + 1,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            latency_ms=round(max(0.0, latency_ms), 3),
            guardrail_triggered=guardrail_triggered,
        )
        self.tools_executed.append(entry)
        return entry

    def elapsed_ms(self) -> float:
        """Return total elapsed execution time in milliseconds."""
        return round((time.perf_counter() - self._start_time) * 1000.0, 3)

    def build_output(
        self,
        provider: str,
        model: str,
        settings: Optional[Settings] = None,
    ) -> TelemetryOutput:
        """Compile the validated TelemetryOutput contract."""
        return TelemetryOutput(
            trace_id=self.trace_id,
            provider=provider,
            model=model,
            latency_ms=self.elapsed_ms(),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            estimated_cost_usd=estimate_cost_usd(
                self.input_tokens,
                self.output_tokens,
                settings=settings,
            ),
            tools_executed=list(self.tools_executed),
            state_transitions=list(self.state_transitions),
        )
