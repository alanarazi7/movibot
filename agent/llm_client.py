"""LLM access for MoviBot, via the course's LLMod.ai OpenAI-compatible endpoint.

This is the only module in the agent that spends money. Everything else --
the catalog, the local E5 embeddings, all three tools -- runs for free, so the
agent can be developed and tested end to end before this is ever called.

There is no mock fallback by design. A silent stand-in that answers without
the model would make a broken configuration look like a working agent, which
is the most expensive kind of bug to find late. Missing credentials raise.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

# Chat model id on LLMod.ai. Confirmed against GET /v1/models, which returns
# exactly two ids for this tenant -- this one and the embedding model. It is
# gpt-5.4-mini, not the gpt-4o-mini previously guessed here, and the "azure/"
# segment is real rather than a mistake. Override with MOVIBOT_MODEL.
DEFAULT_MODEL = "MB5R2CF-azure/gpt-5.4-mini"

# Low but not zero: the loop should make the same tool choices run to run,
# while the final prose stays readable. Not sent to gpt-5 models, which reject
# any temperature other than 1 -- see supports_temperature().
TEMPERATURE = 0.2


def supports_temperature(model: str) -> bool:
    """gpt-5 models accept only temperature=1, and 400 on anything else.

    Sending the default and letting the provider ignore it is not an option:
    LiteLLM raises UnsupportedParamsError rather than dropping the parameter.
    """
    return "gpt-5" not in model.lower()

# One turn's reply is either a few tool calls or a short recommendation.
MAX_TOKENS = 1200


def model_name() -> str:
    return os.environ.get("MOVIBOT_MODEL", DEFAULT_MODEL)


# Values shipped in .env.example and friends. Treating these as configured
# would make a half-set-up deployment look healthy and produce a confusing
# connection error at request time instead of a clear "not configured" one.
_PLACEHOLDER_MARKERS = ("your-", "placeholder", "changeme", "todo", "xxx", "<")


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return not lowered or any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def offline() -> bool:
    """True when spending is disabled outright by MOVIBOT_OFFLINE.

    A hard kill switch, independent of whether credentials exist. Set
    MOVIBOT_OFFLINE=1 to guarantee a run cannot cost anything -- useful for
    tests, demos, and any change to the loop that has not been cost-reviewed.
    """
    return os.environ.get("MOVIBOT_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def is_configured() -> bool:
    """True if a real call could actually succeed. Lets callers degrade politely."""
    if offline():
        return False
    return not _is_placeholder(os.environ.get("OPENAI_API_KEY", ""))


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    # Checked here too, not only in is_configured(), so that no code path can
    # reach the network by calling complete() directly.
    if offline():
        raise RuntimeError(
            "MOVIBOT_OFFLINE is set: model calls are disabled. Unset it to "
            "allow MoviBot to spend budget."
        )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if _is_placeholder(api_key):
        raise RuntimeError(
            "OPENAI_API_KEY is unset or still a placeholder. MoviBot's tools "
            "all run locally at no cost, but composing an answer requires the "
            "LLM. Set real OPENAI_API_KEY and OPENAI_BASE_URL values in .env."
        )

    return OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"))


def complete(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    """One chat completion, optionally offering tools.

    Returns `(message, usage)`. The message is the raw object, so the caller
    can inspect `.tool_calls` and `.content` and append it verbatim to the
    conversation, which the tool-calling protocol requires. The usage counts
    ride alongside because they live on the response, not the message -- an
    earlier version returned the message alone, so every token count in the
    budget block was silently zero.
    """
    model = model_name()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
    }
    if supports_temperature(model):
        kwargs["temperature"] = TEMPERATURE
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = _client().chat.completions.create(**kwargs)
    return response.choices[0].message, usage_of(response)


# How long a budget reading is reused. The figure moves only when this agent
# spends, and a request costs a fraction of a cent, so a minute-old number is
# never misleading -- it just stops a page refresh from hammering the provider.
BUDGET_TTL_SECONDS = 60

_budget_cache: tuple[float, dict[str, Any]] | None = None


def _proxy_root() -> str:
    """LLMod.ai's root, with the OpenAI-compatible /v1 suffix removed.

    The admin routes below (/user/info, /key/info) sit beside /v1 rather than
    under it, so OPENAI_BASE_URL cannot be used as-is.
    """
    base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    return base[:-3].rstrip("/") if base.endswith("/v1") else base


def budget() -> dict[str, Any]:
    """This key's spend against its cap, read live from LLMod.ai.

    Note this does not contradict the module docstring: it is a plain GET
    against the proxy's own accounting, consumes no tokens, and costs nothing.
    LLMod.ai runs LiteLLM, whose /user/info reports `spend` and `max_budget`
    for the calling key, and /key/info adds the key's expiry.

    Returns a dict that is always safe to serialise to the public API: the
    provider's replies also carry the hashed key, the key name, and course and
    group identifiers, none of which are copied out. `configured` is False
    rather than raising when credentials are absent, so the caller can degrade.
    """
    global _budget_cache

    import time

    if _budget_cache is not None:
        fetched_at, cached = _budget_cache
        if time.time() - fetched_at < BUDGET_TTL_SECONDS:
            return {**cached, "cached": True}

    if offline():
        return {"configured": False,
                "reason": "MOVIBOT_OFFLINE is set, so no budget is being spent."}

    root, key = _proxy_root(), os.environ.get("OPENAI_API_KEY", "")
    if not root or _is_placeholder(key):
        return {"configured": False,
                "reason": "OPENAI_API_KEY or OPENAI_BASE_URL is unset."}

    import requests

    headers = {"Authorization": f"Bearer {key}"}

    def get(path: str) -> dict[str, Any]:
        resp = requests.get(root + path, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    try:
        info = get("/user/info").get("user_info") or {}
        spend = float(info.get("spend") or 0.0)
        cap = info.get("max_budget")
        cap = float(cap) if cap is not None else None

        # Best-effort: the expiry is a nice-to-have, and a failure here should
        # not cost the caller the spend figure, which is the point of the call.
        try:
            expires = (get("/key/info").get("info") or {}).get("expires")
        except Exception:  # noqa: BLE001
            expires = None

        result = {
            "configured": True,
            "spend_usd": round(spend, 6),
            "max_budget_usd": cap,
            "remaining_usd": round(cap - spend, 6) if cap is not None else None,
            "used_fraction": round(spend / cap, 6) if cap else None,
            "key_expires": expires,
            "source": "LLMod.ai /user/info (LiteLLM)",
        }
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as JSON
        return {"configured": True, "error": f"{type(exc).__name__}: {exc}"}

    _budget_cache = (time.time(), result)
    return {**result, "cached": False}


def usage_of(response: Any) -> dict[str, int]:
    """Token counts if the provider reported them, else zeros."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }
