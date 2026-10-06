"""Tests for hpccoworklab.ai (HPC-40): config gating, parsing, and fail-safe behavior.
No network - _transport is monkeypatched."""

import pytest

from hpccoworklab import ai
from hpccoworklab.ai import ai_settings, review_script, _parse_findings, AiNotConfigured, AiError


def test_ai_settings_disabled_without_env(monkeypatch):
    monkeypatch.delenv("HPCCOWORKLAB_AI_BASE_URL", raising=False)
    monkeypatch.delenv("HPCCOWORKLAB_AI_API_KEY", raising=False)
    assert ai_settings({})["enabled"] is False


def test_ai_settings_enabled_via_env(monkeypatch):
    monkeypatch.setenv("HPCCOWORKLAB_AI_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("HPCCOWORKLAB_AI_API_KEY", "sk-abc")
    monkeypatch.setenv("HPCCOWORKLAB_AI_MODEL", "custom-model")
    s = ai_settings({})
    assert s["enabled"] is True and s["model"] == "custom-model"


def test_review_script_raises_when_not_configured(monkeypatch):
    monkeypatch.delenv("HPCCOWORKLAB_AI_BASE_URL", raising=False)
    monkeypatch.delenv("HPCCOWORKLAB_AI_API_KEY", raising=False)
    with pytest.raises(AiNotConfigured):
        review_script("#!/bin/bash\n#SBATCH --mem=999T\n")


def test_review_script_returns_findings_from_bullets(monkeypatch):
    monkeypatch.setenv("HPCCOWORKLAB_AI_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("HPCCOWORKLAB_AI_API_KEY", "sk-abc")

    def fake_transport(payload, base_url, api_key, timeout=30):
        return {"choices": [{"message": {"content":
            "- memory request is unrealistically high\n- no output redirection set"}}]}
    monkeypatch.setattr(ai, "_transport", fake_transport)

    findings = review_script("#!/bin/bash\n#SBATCH --mem=999T\n")
    assert len(findings) == 2
    assert "memory" in findings[0].lower()


def test_review_script_json_list(monkeypatch):
    monkeypatch.setenv("HPCCOWORKLAB_AI_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("HPCCOWORKLAB_AI_API_KEY", "sk-abc")
    monkeypatch.setattr(ai, "_transport",
                        lambda *a, **k: {"choices": [{"message": {"content": '["asked 8 GPUs but code uses 1"]'}}]})
    assert review_script("x") == ["asked 8 GPUs but code uses 1"]


def test_review_script_transport_error_raises_aierror(monkeypatch):
    monkeypatch.setenv("HPCCOWORKLAB_AI_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("HPCCOWORKLAB_AI_API_KEY", "sk-abc")

    def boom(*a, **k):
        raise OSError("network down")
    monkeypatch.setattr(ai, "_transport", boom)
    with pytest.raises(AiError):
        review_script("x")


def test_parse_findings_none():
    assert _parse_findings("NONE") == []
    assert _parse_findings("") == []
