"""
hpclint.ai - OPTIONAL, advisory LLM code review (HPC-40).

Strictly additive and fail-safe:
  * Off unless the user passes `--ai` AND configures an endpoint/key.
  * The deterministic checks are always the source of truth; AI output is shown
    as non-authoritative observations and NEVER changes exit codes.
  * No cluster data or secrets are sent - only the job script text, and only to
    a user-configured OpenAI-compatible endpoint (NOT an NRP service).
  * Any failure degrades to "AI skipped: <reason>"; the tool keeps working.
"""

import json
import os
import re
import urllib.request
import urllib.error


class AiNotConfigured(Exception):
    pass


class AiError(Exception):
    pass


def ai_settings(config=None):
    """Resolve endpoint/key/model from env first, then the config `ai:` block."""
    config = config or {}
    ai = config.get("ai") or {}
    base_url = os.environ.get("HPCLINT_AI_BASE_URL") or ai.get("base_url")
    api_key = os.environ.get("HPCLINT_AI_API_KEY") or ai.get("api_key")
    model = os.environ.get("HPCLINT_AI_MODEL") or ai.get("model") or "gpt-4o-mini"
    return {"base_url": base_url, "api_key": api_key, "model": model,
            "enabled": bool(base_url and api_key)}


def _transport(payload, base_url, api_key, timeout=30):
    """Low-level HTTP call to an OpenAI-compatible endpoint (isolated for tests)."""
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _build_messages(script_content, config):
    cluster = (config or {}).get("cluster_name", "the cluster")
    system = (
        "You are a careful HPC reviewer. Given a Slurm job script, list ONLY likely "
        "resource or configuration mistakes (over/under-requesting, wrong queue for the "
        "workload, likely-to-fail settings) as short bullet points. Be concise. If you see "
        "nothing, reply exactly: NONE. Do not restate the script."
    )
    user = f"Cluster: {cluster}\nJob script:\n```\n{script_content}\n```"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _parse_findings(text):
    """Accept a JSON list or bullet/numbered lines; 'NONE' or empty -> []."""
    if not text:
        return []
    t = text.strip()
    if t.upper().startswith("NONE"):
        return []
    try:
        data = json.loads(t)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except (ValueError, TypeError):
        pass
    items = []
    for line in re.split(r"[\r\n]+", t):
        s = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip()
        if s and not s.upper().startswith("NONE"):
            items.append(s)
    return items


def review_script(script_content, config=None, timeout=30):
    """Return a list of advisory findings (possibly empty). Raises
    AiNotConfigured if unset, AiError on transport/parse failure."""
    s = ai_settings(config)
    if not s["enabled"]:
        raise AiNotConfigured(
            "AI not configured (set HPCLINT_AI_BASE_URL and HPCLINT_AI_API_KEY, "
            "or an `ai:` block in the config).")
    payload = {"model": s["model"], "messages": _build_messages(script_content, config),
               "temperature": 0}
    try:
        resp = _transport(payload, s["base_url"], s["api_key"], timeout=timeout)
    except (urllib.error.URLError, ValueError, OSError) as e:
        raise AiError(str(e))
    try:
        content = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise AiError("unexpected AI response shape")
    return _parse_findings(content)
