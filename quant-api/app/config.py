from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="Cross-Market Valuation API")
    app_env: str = Field(default="dev")
    database_url: str = Field(
        default="postgresql+psycopg://quant:quant@localhost:5432/quant_platform",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    raw_archive_path: Path = Field(default=Path("./raw_archives"), alias="RAW_ARCHIVE_PATH")
    event_trigger_threshold: float = Field(default=0.05, alias="EVENT_TRIGGER_THRESHOLD")
    live_trading_enabled: bool = Field(default=False, alias="LIVE_TRADING_ENABLED")
    real_data_only: bool = Field(default=True, alias="REAL_DATA_ONLY")
    universe_tickers: str = Field(
        default="TNK,INSW,STNG,SBLK,DHT,FRO,NAT,SM,MTDR,RRC,MUR,AR", alias="UNIVERSE_TICKERS"
    )
    shipping_proxy_tickers: str = Field(default="BDRY,BOAT,SEA", alias="SHIPPING_PROXY_TICKERS")
    commodity_proxy_tickers: str = Field(default="BZ=F,CL=F", alias="COMMODITY_PROXY_TICKERS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Execution layer
    discord_webhook_url: str = Field(default="", alias="DISCORD_WEBHOOK_URL")
    ib_host: str = Field(default="127.0.0.1", alias="IB_HOST")
    ib_port: int = Field(default=7497, alias="IB_PORT")
    ib_client_id_sync: int = Field(default=1, alias="IB_CLIENT_ID_SYNC")
    ib_client_id_exec: int = Field(default=2, alias="IB_CLIENT_ID_EXEC")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def universe(self) -> List[str]:
        return [x.strip().upper() for x in self.universe_tickers.split(",") if x.strip()]

    @property
    def shipping_proxies(self) -> List[str]:
        return [x.strip().upper() for x in self.shipping_proxy_tickers.split(",") if x.strip()]

    @property
    def commodity_proxies(self) -> List[str]:
        return [x.strip().upper() for x in self.commodity_proxy_tickers.split(",") if x.strip()]

    @property
    def effective_database_url(self) -> str:
        # Keep Postgres+Timescale as the target. Fall back to SQLite only when psycopg
        # is missing so local tests can run without native driver setup.
        if self.database_url.startswith("postgresql") and importlib.util.find_spec("psycopg") is None:
            return "sqlite+pysqlite:///./quant_platform_local.db"
        return self.database_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.raw_archive_path.mkdir(parents=True, exist_ok=True)
    return settings
