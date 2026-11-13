from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import features as features_settings
from . import headroom_settings

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CONFIG_PATH = DATA_DIR / "config.json"
USAGE_DB_PATH = DATA_DIR / "usage.db"
SIDECAR_LOG_PATH = DATA_DIR / "sidecar.log"
LEDGER_DIR = DATA_DIR / "sessions"
DASHBOARD_DIR = PROJECT_DIR / "dashboard"

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 18090
DEFAULT_UPSTREAM_URL = "http://127.0.0.1:18080"


@dataclass
class Settings:
    upstream_url: str = DEFAULT_UPSTREAM_URL
    api_key: str = ""
    exposed_models: list[str] = field(default_factory=list)
    route_overrides: dict[str, str] = field(default_factory=dict)
    headroom: dict[str, Any] = field(default_factory=headroom_settings.defaults)
    features: dict[str, Any] = field(default_factory=features_settings.defaults)

    def effective_key(self) -> str:
        return os.environ.get("TOKEN_PROXY_UPSTREAM_KEY") or self.api_key

    def proxy_url(self) -> str:
        return f"http://{LISTEN_HOST}:{LISTEN_PORT}"

    def sidecar_url(self) -> str:
        return f"http://{LISTEN_HOST}:{self.headroom.get('sidecar_port', 8787)}"


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    settings = Settings()
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        settings.upstream_url = raw.get("upstream_url", settings.upstream_url)
        settings.api_key = raw.get("api_key", "")
        settings.exposed_models = list(raw.get("exposed_models", []))
        settings.route_overrides = dict(raw.get("route_overrides", {}))
        stored = {
            key: value for key, value in dict(raw.get("headroom", {})).items()
            if key in {e["key"] for e in headroom_settings.SCHEMA}
        }
        settings.headroom = headroom_settings.normalize(stored)
        stored_features = {
            key: value for key, value in dict(raw.get("features", {})).items()
            if key in {e["key"] for e in features_settings.FEATURES}
        }
        settings.features = features_settings.normalize(stored_features)
    return settings


def save_settings(settings: Settings, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
