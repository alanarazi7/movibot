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

# Chat model id on LLMod.ai. Their ids are namespaced (the embedding model is
# "MB5R2CF-azure/text-embedding-3-small"), so the chat id follows the same
# shape. Override with MOVIBOT_MODEL if the deployment differs.
DEFAULT_MODEL = "MB5R2CF-azure/gpt-4o-mini"

# Low but not zero: the loop should make the same tool choices run to run,
# while the final prose stays readable.
TEMPERATURE = 0.2

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

    Returns the raw `message` object so the caller can inspect `.tool_calls`
    and `.content` and append it verbatim to the conversation, which the
    tool-calling protocol requires.
    """
    kwargs: dict[str, Any] = {
        "model": model_name(),
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = _client().chat.completions.create(**kwargs)
    return response.choices[0].message


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
