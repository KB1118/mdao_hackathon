import json

from proxy import claude_config


def test_apply_then_restore_returns_original_bytes(tmp_path):
    path = tmp_path / "settings.json"
    original = '{\n  "env": {\n    "ANTHROPIC_BASE_URL": "http://127.0.0.1:18080"\n  },\n  "theme": "dark"\n}\n'
    path.write_text(original, encoding="utf-8")

    status = claude_config.apply(path, "http://127.0.0.1:18090", ["claude-haiku-5-5"])
    assert status["base_url"] == "http://127.0.0.1:18090"
    assert status["backup_exists"] is True

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["availableModels"] == ["claude-haiku-5-5"]
    assert data["theme"] == "dark"

    claude_config.restore(path)
    assert path.read_text(encoding="utf-8") == original
    assert not claude_config.backup_path(path).exists()


def test_second_apply_keeps_original_backup(tmp_path):
    path = tmp_path / "settings.json"
    original = '{"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:18080"}}'
    path.write_text(original, encoding="utf-8")

    claude_config.apply(path, "http://127.0.0.1:18090", [])
    claude_config.apply(path, "http://127.0.0.1:18090", ["claude-haiku-5-5"])
    claude_config.restore(path)

    assert path.read_text(encoding="utf-8") == original


def test_restore_removes_file_that_apply_created(tmp_path):
    path = tmp_path / "settings.json"

    claude_config.apply(path, "http://127.0.0.1:18090", [])
    assert path.exists()

    claude_config.restore(path)
    assert not path.exists()
