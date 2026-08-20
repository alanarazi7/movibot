#!/usr/bin/env python3
"""💰 Runs one query through the real loop and reports what it did, compactly.

    python scripts/run_case.py "A movie with a cat that wears a hat"

Prints the trace shape, the token cost, and the answer -- enough to judge a
case without reading a 30 KB JSON blob. Writes the full result beside it so the
detail is there when a summary is not enough.

This spends. Every invocation is one real request against the course
deployment, so it takes exactly one query and never loops over a list.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

OUT = _ROOT / "artifacts"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    prompt = sys.argv[1]

    from agent import llm_client, loop

    before = llm_client.budget().get("spend_usd")
    started = time.time()
    result = loop.execute(prompt)
    elapsed = time.time() - started

    steps = result.get("steps") or []
    model_steps = [s for s in steps if s["module"] in ("Planner", "Observer")]
    prompt_tok = sum((s.get("usage") or {}).get("prompt_tokens", 0) for s in model_steps)
    out_tok = sum((s.get("usage") or {}).get("completion_tokens", 0) for s in model_steps)
    cached = sum((s.get("usage") or {}).get("cached_tokens", 0) for s in model_steps)

    print(f"\n\033[1m{prompt}\033[0m")
    print(f"status {result['status']}  ·  {elapsed:.1f}s  ·  "
          f"{len(model_steps)} model calls  ·  {prompt_tok:,} in "
          f"({cached:,} cached)  ·  {out_tok:,} out")

    print("\ntrace")
    for s in steps:
        mod = s["module"]
        if mod == "Planner":
            calls = [c["name"] for c in (s["response"].get("tool_calls") or [])]
            note = ", ".join(calls) if calls else "(no tool call -- this is the answer)"
            print(f"   Planner        -> {note}")
        elif mod == "Observer":
            f = s["response"].get("findings") or []
            verdicts = ", ".join(f"{x.get('film')}={x.get('verdict')}" for x in f)
            rejected = sum(1 for x in f if x.get("quote_rejected"))
            print(f"   Observer       -> {verdicts or s['response'].get('error')}"
                  + (f"   [{rejected} quote(s) rejected]" if rejected else ""))
        else:
            args = {k: v for k, v in (s["prompt"].get("arguments") or {}).items()}
            print(f"   {mod:14} <- {json.dumps(args, ensure_ascii=False)[:118]}")
            if s.get("scope"):
                print(f"   {'':14}    scope now: {s['scope']}")

    print("\nanswer")
    for line in (result.get("response") or result.get("error") or "").splitlines():
        print("   " + line)

    after = llm_client.budget.__wrapped__() if hasattr(llm_client.budget, "__wrapped__") else None
    OUT.mkdir(exist_ok=True)
    slug = "".join(c if c.isalnum() else "-" for c in prompt.lower())[:60].strip("-")
    path = OUT / f"{slug}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nfull trace: {path.relative_to(_ROOT)}   (spend before this run: ${before})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
