#!/usr/bin/env python3
"""Reports which credentials are usable, and what each one unlocks.

Costs nothing and makes no network call by default -- it only inspects the
environment and the installed packages. Never prints a secret: values are
reported by length and shape only.

    python scripts/check_credentials.py
    python scripts/check_credentials.py --ping    # 💰 one tiny model call

The configured model id is checked against GET /v1/models, which lists what the
key may actually call and consumes no tokens. --ping goes further and sends a
few tokens through the real completion path, for end-to-end proof; it spends, so
it is opt-in.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

OK, BAD, WARN = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m!\033[0m"

PLACEHOLDER = re.compile(r"your-|todo|placeholder|xxx|^<", re.I)


def looks_placeholder(value: str) -> bool:
    return not value.strip() or bool(PLACEHOLDER.search(value))


def check(name: str, pattern: str, why: str) -> bool:
    """Report one variable by shape. Never prints the value."""
    value = os.environ.get(name, "")
    if not value:
        print(f"  {BAD} {name:22} unset            {why}")
        return False
    if looks_placeholder(value):
        print(f"  {BAD} {name:22} placeholder      {why}")
        return False
    if pattern and not re.match(pattern, value):
        print(f"  {WARN} {name:22} set ({len(value)} chars), but does not match the expected shape")
        print(f"    {'':22} {why}")
        return False
    print(f"  {OK} {name:22} set ({len(value)} chars)")
    return True


def list_models() -> list[str] | None:
    """Model ids this key can actually use, or None if the call failed.

    Free -- listing models consumes no tokens -- and strictly better than
    guessing an id's shape: it proves the key authenticates *and* enumerates
    what it may call. This is how the gpt-4o-mini/gpt-5.4-mini question was
    settled, after inspection alone had guessed wrong twice.
    """
    import requests

    base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not base or not key:
        return None
    url = base + ("/models" if base.endswith("/v1") else "/v1/models")
    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=30)
        if not resp.ok:
            return None
        return sorted(m.get("id", "") for m in resp.json().get("data", []))
    except Exception:
        return None


def ping() -> None:
    """One minimal completion, to prove key + base URL + model id agree.

    Goes through llm_client.complete() rather than building a client here, so
    it exercises exactly the path the agent uses -- including the offline guard
    and the placeholder-credential check.
    """
    from agent import llm_client

    model = llm_client.model_name()
    print(f"\n💰 Pinging {model} ...")
    try:
        message, usage = llm_client.complete(
            [{"role": "user", "content": "Reply with the single word: ok"}]
        )
        print(f"  {OK} model responded: {(message.content or '').strip()!r}")
        print(f"    tokens: {usage['prompt_tokens']} in, {usage['completion_tokens']} out")
        print("\n  The model id is correct. The paid track is unblocked.")
    except Exception as exc:
        print(f"  {BAD} {type(exc).__name__}: {exc}")
        print("\n  A 404 here means the id is wrong rather than the key -- but the id was")
        print("  already checked against the tenant's model list above, so this more likely")
        print("  indicates a quota, permission, or network problem.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ping", action="store_true",
                        help="💰 make one tiny model call to verify the model id")
    args = parser.parse_args()

    print("\nRequired to answer any query")
    llm_ok = all([
        check("OPENAI_API_KEY", r"^\S{20,}$", "LLMod.ai API key"),
        check("OPENAI_BASE_URL", r"^https?://", "LLMod.ai base URL"),
    ])

    from agent import llm_client
    model = llm_client.model_name()
    available = list_models() if llm_ok else None

    if available is None:
        print(f"  {WARN} {'MOVIBOT_MODEL':22} {model}  (could not list models to check)")
    elif model in available:
        print(f"  {OK} {'MOVIBOT_MODEL':22} {model}")
    else:
        llm_ok = False
        print(f"  {BAD} {'MOVIBOT_MODEL':22} {model}")
        print(f"    {'':22} not offered by this tenant. Available:")
        for m in available:
            print(f"    {'':22}   {m}")

    print("\nOptional -- only for the cloud backends")
    pine = all([
        check("PINECONE_API_KEY", r"^\S{20,}$", "only needed for MOVIBOT_EMBEDDINGS=cloud"),
        check("PINECONE_INDEX_NAME", r"^\S+$", "index name, e.g. movibot-plots"),
    ])
    supa = all([
        check("SUPABASE_URL", r"^https://[a-z0-9]+\.supabase\.co/?$", "https://<ref>.supabase.co"),
        check("SUPABASE_KEY", r"^eyJ[\w-]{20,}", "a JWT starting eyJ, usually 200+ chars"),
    ])

    print("\nLocal backends")
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
        local_embed = True
        print(f"  {OK} {'sentence-transformers':22} installed -- local semantic search works")
    except ImportError:
        local_embed = False
        print(f"  {BAD} {'sentence-transformers':22} missing -- pip install -r requirements-local.txt")

    offline = os.environ.get("MOVIBOT_OFFLINE", "").strip().lower() in ("1", "true", "yes")
    if offline:
        print(f"  {WARN} {'MOVIBOT_OFFLINE':22} set -- all spending is blocked at the client")

    print("\nWhat this configuration can do")
    if llm_ok and local_embed and not offline:
        print(f"  {OK} run the 11 test cases locally, using local catalog and local embeddings")
        print("    nothing else is needed for that -- Pinecone and Supabase are not involved")
    elif llm_ok and offline:
        print(f"  {WARN} credentials look usable, but MOVIBOT_OFFLINE blocks every model call")
    elif llm_ok:
        print(f"  {WARN} the planner can run, but semantic search needs either")
        print("    requirements-local.txt (free) or Pinecone (paid)")
    else:
        print(f"  {BAD} cannot answer a query yet -- the two required values above are missing")

    print(f"  {OK if pine else BAD} Pinecone path (MOVIBOT_EMBEDDINGS=cloud), required for production")
    print(f"  {OK if supa else BAD} Supabase path (MOVIBOT_BACKEND=cloud)")

    if not args.ping:
        print("\nThe model id above was checked against the tenant's live model list,")
        print("which costs nothing. --ping additionally sends a few tokens through the")
        print("real completion path, if you want end-to-end proof:")
        print("    python scripts/check_credentials.py --ping     # 💰 a fraction of a cent")
    else:
        ping()
    print()


if __name__ == "__main__":
    main()
