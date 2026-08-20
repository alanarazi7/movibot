#!/usr/bin/env python3
"""💰 Runs the test bed and writes one compact summary per case.

    python scripts/run_suite.py            # every case
    python scripts/run_suite.py 1 3 5      # only these, 1-indexed

The cases live in public/index.html so the page and this script cannot drift:
there is one list, and it is the one a reviewer clicks. Full traces go to
artifacts/ and a summary table to artifacts/suite.md, which is what the
expectations get rewritten against.

Every case is one real request. Ten cases cost a few cents.
"""

from __future__ import annotations

import json
import re
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


def cases() -> list[str]:
    src = (_ROOT / "public" / "index.html").read_text()
    blk = src[src.index("const TEST_CASES = ["):]
    blk = blk[:blk.index("\n    ];")]
    return [q.replace("\\'", "'").replace("\\\\", "\\")
            for q in re.findall(r"q: '((?:[^'\\]|\\.)*)'", blk)]


def summarise(prompt: str, result: dict, elapsed: float) -> dict:
    steps = result.get("steps") or []
    model = [s for s in steps if s["module"] in ("Planner", "Observer")]
    tools_used = [s["module"] for s in steps if s["module"] not in ("Planner", "Observer")]
    return {
        "prompt": prompt,
        "status": result.get("status"),
        "seconds": round(elapsed, 1),
        "model_calls": len(model),
        "prompt_tokens": sum((s.get("usage") or {}).get("prompt_tokens", 0) for s in model),
        "cached_tokens": sum((s.get("usage") or {}).get("cached_tokens", 0) for s in model),
        "tools": tools_used,
        "tool_args": [
            {k: v for k, v in (s["prompt"].get("arguments") or {}).items()}
            for s in steps if s["module"] not in ("Planner", "Observer")
        ],
        "observer": [
            {f.get("film"): f.get("verdict") for f in (s["response"].get("findings") or [])}
            for s in steps if s["module"] == "Observer"
        ],
        "response": result.get("response") or result.get("error"),
    }


def main() -> int:
    wanted = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else None
    all_cases = cases()
    picked = [(i, q) for i, q in enumerate(all_cases, 1) if wanted is None or i in wanted]

    from agent import llm_client, loop

    before = llm_client.budget().get("spend_usd")
    OUT.mkdir(exist_ok=True)
    summaries = []

    for i, prompt in picked:
        started = time.time()
        result = loop.execute(prompt)
        s = summarise(prompt, result, time.time() - started)
        s["case"] = i
        summaries.append(s)
        (OUT / f"case-{i:02d}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[{i:2}] {s['status']:5} {s['seconds']:>5.1f}s  "
              f"{s['model_calls']} calls  {' -> '.join(s['tools']) or '(no tools)'}")
        print(f"     {(s['response'] or '')[:150].replace(chr(10), ' ')}")

    (OUT / "suite.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    after = llm_client.budget()
    print(f"\nspend {before} -> {after.get('spend_usd')}  "
          f"({len(picked)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
