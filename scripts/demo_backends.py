#!/usr/bin/env python3
"""Quick demo: Compare IDF vs. E5 embedding backends side-by-side.

Usage:
    python scripts/demo_backends.py

No arguments needed; compares both backends on sample queries.
"""

import subprocess
import sys


QUERIES = [
    "Find me an animated adventure about a magical kingdom with a strong female lead",
    "heartwarming adventure with a loyal animal companion",
    "a fun family movie with talking animals",
]


def test_backend(backend_name, env_var_value):
    """Test a backend via subprocess to isolate environment."""
    results = {}

    for query in QUERIES:
        code = f"""
import os
os.environ["PLOT_SEARCH_BACKEND"] = "{env_var_value}"

from agent.tools import plot_search

results = plot_search.run("{query}", top_k=3)
for r in results:
    print(f"{{r['title']}} ({{r['release_year']}}) - {{r['score']:.4f}}")
"""

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True
        )

        results[query] = result.stdout.strip().split('\n')

    return results


def main():
    print("=" * 80)
    print("BACKEND COMPARISON: IDF vs. E5 Embeddings")
    print("=" * 80)

    print("\n📊 Testing IDF backend (keyword-based)...")
    idf_results = test_backend("idf", "idf")

    print("📊 Testing E5 backend (semantic)...")
    embed_results = test_backend("embedding", "embedding")

    # Display side-by-side
    for i, query in enumerate(QUERIES, 1):
        print("\n" + "=" * 80)
        print(f"Query {i}: {query}")
        print("=" * 80)

        print("\n[IDF Backend - keyword matching]")
        for line in idf_results[query]:
            if line.strip():
                print(f"  {line}")

        print("\n[E5 Backend - semantic search]")
        for line in embed_results[query]:
            if line.strip():
                print(f"  {line}")

        print("\n💡 Observation:")
        print("   IDF backend matches exact keywords (fast, structured)")
        print("   E5 backend finds thematic similarity (slow, flexible)")


if __name__ == "__main__":
    main()
