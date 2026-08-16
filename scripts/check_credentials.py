#!/usr/bin/env python3
"""Reports which credentials are usable, and what each one unlocks.

Costs nothing and makes no network call by default -- it only inspects the
environment and the installed packages. Never prints a secret: values are
reported by length and shape only.

    python scripts/check_credentials.py
    python scripts/check_credentials.py --ping    # 💰 one tiny model call

--ping is the only way to settle the open question of whether the model id is
right, since a wrong id fails at request time rather than at configuration
time. It sends a handful of tokens and costs a fraction of a cent, but it does
spend, so it is opt-in.
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
        message = llm_client.complete(
            [{"role": "user", "content": "Reply with the single word: ok"}]
        )
        print(f"  {OK} model responded: {(message.content or '').strip()!r}")
        print("\n  The model id is correct. The paid track is unblocked.")
    except Exception as exc:
        print(f"  {BAD} {type(exc).__name__}: {exc}")
        print("\n  If this is a 404 or 'model not found', the id is wrong rather than the key.")
        print("  The sibling medium-rag-hw project uses <TENANT>-<model> with no azure/")
        print("  segment, e.g. MB5R2CF-gpt-5-mini. Override with MOVIBOT_MODEL.")


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
    unusual = "azure/" in model or not re.match(r"^[A-Z0-9]+-", model)
    print(f"  {WARN if unusual else OK} {'MOVIBOT_MODEL':22} {model}")
    if unusual:
        print(f"    {'':22} expected <TENANT>-<model> with no azure/ segment;")
        print(f"    {'':22} verify with --ping before assuming this works")

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
        print("\nA valid-looking key is not a working one. Confirm the model id with:")
        print("    python scripts/check_credentials.py --ping     # 💰 a fraction of a cent")
    else:
        ping()
    print()


if __name__ == "__main__":
    main()
