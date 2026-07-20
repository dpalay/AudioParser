"""Send a transcript to an OpenAI-compatible chat API for meeting analysis.

Designed for org LLM proxies: point --base-url (or AUDIOPARSER_LLM_BASE_URL /
OPENAI_BASE_URL) at the proxy and supply the key via AUDIOPARSER_LLM_API_KEY /
OPENAI_API_KEY. stdlib-only, no SDK dependency.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PROMPT = """\
You are analyzing a meeting transcript. Speaker names and timestamps are
accurate. Produce:

1. **Summary** - 3-5 sentences on what the meeting was about and how it went.
2. **Decisions** - every decision made, with who made or approved it.
3. **Action items** - a list of `owner - task - deadline (if mentioned)`.
4. **Disagreements / open questions** - unresolved points and who holds
   which position.
5. **Follow-ups** - anything explicitly deferred to a later meeting.

Use the speaker names from the transcript. If something is ambiguous, say so
rather than guessing."""

SYSTEM_PROMPT = "You are a precise meeting analyst."


def resolve_config(
    base_url: str | None, model: str | None, api_key: str | None
) -> tuple[str, str, str]:
    base_url = (
        base_url
        or os.environ.get("AUDIOPARSER_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
    )
    model = model or os.environ.get("AUDIOPARSER_LLM_MODEL")
    api_key = (
        api_key
        or os.environ.get("AUDIOPARSER_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    missing = [
        name
        for name, value in [
            ("base URL (--base-url or AUDIOPARSER_LLM_BASE_URL)", base_url),
            ("model (--model or AUDIOPARSER_LLM_MODEL)", model),
            ("API key (AUDIOPARSER_LLM_API_KEY or OPENAI_API_KEY)", api_key),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError("Missing LLM settings: " + "; ".join(missing))
    return base_url.rstrip("/"), model, api_key


def build_request(
    transcript: str, base_url: str, model: str, api_key: str, prompt: str
) -> urllib.request.Request:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{prompt}\n\n---\n\n{transcript}",
            },
        ],
    }
    return urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )


def extract_reply(response_body: bytes | str) -> str:
    if isinstance(response_body, bytes):
        response_body = response_body.decode()
    data = json.loads(response_body)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected API response shape: {response_body[:500]}"
        ) from exc


def analyze_transcript(
    transcript_path: str | Path,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    prompt: str | None = None,
    timeout: float = 300.0,
) -> str:
    transcript = Path(transcript_path).read_text()
    base_url, model, api_key = resolve_config(base_url, model, api_key)
    request = build_request(transcript, base_url, model, api_key, prompt or DEFAULT_PROMPT)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return extract_reply(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"LLM API returned HTTP {exc.code}: {detail}") from exc
