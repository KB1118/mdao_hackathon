from __future__ import annotations

import re

ROUTE_ANTHROPIC = "anthropic"
ROUTE_RESPONSES = "responses"
VALID_ROUTES = {ROUTE_ANTHROPIC, ROUTE_RESPONSES}


def normalize_model(name: str) -> str:
    """Strip Claude Code's context suffix, e.g. 'claude-sonnet-5-5[1m]'."""
    return re.sub(r"\[[^\]]*\]$", "", name or "")


def route_for(model: str, overrides: dict[str, str]) -> str:
    if model in overrides:
        return overrides[model]
    return ROUTE_RESPONSES if model.startswith("gpt-") else ROUTE_ANTHROPIC


def upstream_base(url: str) -> str:
    url = url.strip().rstrip("/")
    return url[:-3] if url.endswith("/v1") else url
