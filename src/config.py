"""Application configuration managed via Pydantic Settings."""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General application settings
    app_name: str = "iFood Incident Triage Agent"
    app_env: str = "development"
    log_level: str = "INFO"

    # LLM provider settings
    llm_provider: Literal["gemini", "openai", "mock"] = "gemini"
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # Financial and operational guardrails
    max_auto_coupon_brl: float = Field(default=20.0, gt=0.0)
    max_auto_refund_brl: float = Field(default=50.0, gt=0.0)
    high_value_order_threshold_brl: float = Field(default=100.0, gt=0.0)
    max_delivered_minutes_for_auto_refund: int = Field(default=60, ge=0)

    # SQLite persistence
    sqlite_db_path: str = ":memory:"

    # Telemetry cost estimation (USD per 1M tokens)
    input_cost_per_1m_usd: float = Field(default=0.15, ge=0.0)
    output_cost_per_1m_usd: float = Field(default=0.60, ge=0.0)

    @property
    def active_model(self) -> str:
        """Return the model identifier for the active LLM provider."""
        if self.llm_provider == "gemini":
            return self.gemini_model
        if self.llm_provider == "openai":
            return self.openai_model
        return "mock-deterministic-v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
