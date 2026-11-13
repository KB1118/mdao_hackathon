import json

from proxy.session_key import extract_session

ALL = {"claude_header": True, "claude_metadata": True, "codex_header": True, "codex_prompt_cache_key": True}


def test_claude_header_is_used_and_namespaced():
    body = {"system": "sys", "tools": [{"name": "Read"}], "messages": []}
    found = extract_session({"X-Claude-Code-Session-Id": "abc-123"}, body, ALL)
    assert found.source == "claude_header"
    assert found.key.startswith("claude:abc-123:")


def test_claude_metadata_gives_same_key_as_header():
    body = {
        "metadata": {"user_id": json.dumps({"device_id": "d", "session_id": "abc-123"})},
        "system": "sys",
        "tools": [{"name": "Read"}],
    }
    from_header = extract_session({"x-claude-code-session-id": "abc-123"}, body, ALL)
    from_metadata = extract_session({}, body, ALL)
    assert from_header.key == from_metadata.key
    assert from_metadata.source == "claude_metadata"


def test_subagent_has_its_own_key_under_the_same_session():
    parent = {"system": "A" * 9000, "tools": [{"name": "Read"}, {"name": "Task"}]}
    subagent = {"system": "B" * 3000, "tools": [{"name": "Read"}]}
    headers = {"x-claude-code-session-id": "s1"}
    assert extract_session(headers, parent, ALL).key != extract_session(headers, subagent, ALL).key


def test_codex_header_and_prompt_cache_key_give_same_key():
    body = {"instructions": "x", "tools": [], "prompt_cache_key": "019f-abc"}
    from_header = extract_session({"session-id": "019f-abc"}, body, ALL)
    from_body = extract_session({}, body, ALL)
    assert from_header.key == from_body.key
    assert from_header.source == "codex_header"
    assert from_body.source == "codex_prompt_cache_key"


def test_disabled_sources_are_skipped():
    enabled = {**ALL, "claude_header": False}
    assert extract_session({"x-claude-code-session-id": "abc"}, {}, enabled) is None


def test_invalid_ids_are_rejected():
    assert extract_session({"x-claude-code-session-id": "bad value"}, {}, ALL) is None
    assert extract_session({"x-claude-code-session-id": "a" * 300}, {}, ALL) is None


def test_no_signal_returns_none():
    assert extract_session({"user-agent": "curl"}, {"messages": []}, ALL) is None
