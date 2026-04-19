from __future__ import annotations
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class ElectionSettings(BaseSettings):
    election_db_path: str = Field(
        default="C:/Users/BrettC/OneDrive/Documents/election_arb.db",
        alias="ELECTION_DB_PATH",
    )
    market_poll_interval_seconds: int = Field(default=300, alias="ELECTION_POLL_INTERVAL")
    altdata_poll_interval_seconds: int = Field(default=3600, alias="ELECTION_ALTDATA_INTERVAL")
    discord_webhook_url: str = Field(default="", alias="ELECTION_DISCORD_WEBHOOK")
    arb_alert_min_edge_pct: float = Field(default=1.0, alias="ELECTION_ARB_MIN_EDGE")
    alpha_alert_min_confidence: float = Field(default=0.7, alias="ELECTION_ALPHA_MIN_CONF")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_election_settings() -> ElectionSettings:
    return ElectionSettings()
