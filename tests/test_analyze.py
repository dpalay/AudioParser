import json

import pytest

from audioparser.analyze import (
    DEFAULT_PROMPT,
    build_request,
    extract_reply,
    resolve_config,
)


def test_resolve_config_from_env(monkeypatch):
    monkeypatch.setenv("AUDIOPARSER_LLM_BASE_URL", "https://proxy.example.com/v1/")
    monkeypatch.setenv("AUDIOPARSER_LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("AUDIOPARSER_LLM_API_KEY", "sk-test")
    base_url, model, key = resolve_config(None, None, None)
    assert base_url == "https://proxy.example.com/v1"  # trailing slash stripped
    assert model == "gpt-5.4"
    assert key == "sk-test"


def test_resolve_config_openai_fallback(monkeypatch):
    for var in ("AUDIOPARSER_LLM_BASE_URL", "AUDIOPARSER_LLM_API_KEY",
                "AUDIOPARSER_LLM_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://x.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-2")
    base_url, model, key = resolve_config(None, "m", None)
    assert (base_url, model, key) == ("https://x.example.com/v1", "m", "sk-2")


def test_resolve_config_missing_raises(monkeypatch):
    for var in ("AUDIOPARSER_LLM_BASE_URL", "AUDIOPARSER_LLM_API_KEY",
                "AUDIOPARSER_LLM_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="Missing LLM settings"):
        resolve_config(None, None, None)


def test_build_request_shape():
    req = build_request(
        "**Dave** [00:01]: hello", "https://p.example.com/v1", "gpt-5.4", "sk", DEFAULT_PROMPT
    )
    assert req.full_url == "https://p.example.com/v1/chat/completions"
    assert req.get_header("Authorization") == "Bearer sk"
    body = json.loads(req.data)
    assert body["model"] == "gpt-5.4"
    assert body["messages"][0]["role"] == "system"
    user = body["messages"][1]["content"]
    assert "Action items" in user and "**Dave** [00:01]: hello" in user


def test_extract_reply():
    body = json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": "the summary"}}]}
    )
    assert extract_reply(body) == "the summary"


def test_extract_reply_bad_shape():
    with pytest.raises(RuntimeError, match="Unexpected API response"):
        extract_reply(json.dumps({"error": "nope"}))
