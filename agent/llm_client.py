"""LLMod.ai client wrapper - OpenAI-compatible, mirrors medium-rag-hw's app.py.

TODO (next pass, after skeleton review): nothing in this module is called
yet. app.py currently returns hardcoded stub data for /api/execute.
"""

import os

CHAT_MODEL = "MB5R2CF-azure/gpt-5.4-mini"
EMBEDDING_MODEL = "MB5R2CF-azure/text-embedding-3-small"


def get_client():
    """Returns an OpenAI-compatible client pointed at LLMod.ai.

    TODO: wire this into agent/react_loop.py once the skeleton is reviewed.
    Requires OPENAI_API_KEY / OPENAI_BASE_URL to be set (see .env.example).
    """
    from openai import OpenAI  # local import: keep this an unused dependency until wired in

    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
