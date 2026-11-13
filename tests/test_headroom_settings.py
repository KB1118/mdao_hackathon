from pathlib import Path

import headroom
import pytest

from proxy import headroom_settings as hs


def _installed_source() -> str:
    root = Path(headroom.__file__).parent
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in root.rglob("*.py"))


def test_every_env_name_exists_in_installed_headroom():
    source = _installed_source()
    for entry in hs.SCHEMA:
        if entry["kind"] == "env" and entry["key"] != "HF_HUB_OFFLINE":
            assert entry["key"] in source, f"{entry['key']} not found in headroom-ai"


def test_defaults_round_trip_through_normalize():
    defaults = hs.defaults()
    assert hs.normalize(defaults) == defaults


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        hs.normalize({"retries": -1})
    with pytest.raises(ValueError):
        hs.normalize({"HEADROOM_MODE": "turbo"})
    with pytest.raises(ValueError):
        hs.normalize({"not_a_setting": 1})


def test_sidecar_env_maps_bools_and_skips_empty():
    values = hs.normalize({"HEADROOM_OFFLINE": True, "HEADROOM_TOOL_DESC_MAX_CHARS": 200})
    env = hs.sidecar_env(values)
    assert env["HEADROOM_OFFLINE"] == "1"
    assert env["HEADROOM_TOOL_DESC_MAX_CHARS"] == "200"
    assert env["HEADROOM_BEACON"] == "off"
    assert "HEADROOM_PROTECT_TOOL_RESULTS" not in env


def test_request_config_carries_session_and_mode():
    config = hs.request_config(hs.normalize({"config_mode": "lossless", "target_ratio": 0.5}), "claude:s:f")
    assert config["session_id"] == "claude:s:f"
    assert config["mode"] == "lossless"
    assert config["target_ratio"] == 0.5


def test_unset_mode_is_omitted_from_request():
    config = hs.request_config(hs.normalize({"config_mode": "unset"}), "claude:s:f")
    assert "mode" not in config


def test_gateway_block_never_requests_redrive():
    block = hs.gateway_block(hs.normalize({"can_redrive": True}))
    assert block["can_redrive"] is False
