"""Interactive Streamlit demo for E5 embedding backend testing."""

import os
import sys
import json
import subprocess
from typing import Any

import streamlit as st
import pandas as pd


# Page config
st.set_page_config(
    page_title="MoviBot: PlotSearch Backends",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 MoviBot: PlotSearch Backend Demo")
st.markdown("""
Compare **IDF keyword matching** vs. **E5 semantic embeddings** for movie search.
""")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

backend_choice = st.sidebar.radio(
    "Select backend:",
    options=["E5 Embedding (Semantic)", "IDF (Keyword Matching)", "Compare Both"],
    help="E5: finds thematic similarity even without exact keywords\nIDF: matches exact words in plot text"
)

# Map choice to env var
backend_env = {
    "E5 Embedding (Semantic)": "embedding",
    "IDF (Keyword Matching)": "idf",
    "Compare Both": "both"
}[backend_choice]

top_k = st.sidebar.slider(
    "Number of results:",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

show_full_trace = st.sidebar.checkbox("Show full ReAct trace", value=False)

# Sample queries
st.sidebar.header("📝 Sample Queries")
sample_queries = [
    "Find me an animated adventure about a magical kingdom with a strong female lead",
    "heartwarming adventure with a loyal animal companion",
    "a fun family movie with talking animals",
    "Find me a Disney movie from the 1990s with no deaths, safe for a toddler",
    "magical adventure involving love and music",
]

query_choice = st.sidebar.selectbox(
    "Or choose a sample:",
    options=["Custom query"] + sample_queries,
)

# Main query input
if query_choice == "Custom query":
    user_query = st.text_input(
        "Enter your query:",
        placeholder="e.g., 'a heartwarming adventure with animals'",
        key="custom_query"
    )
else:
    user_query = query_choice
    st.info(f"📌 Using sample query: **{query_choice}**")

if not user_query:
    st.info("👆 Enter a query to get started!")
    st.stop()

st.divider()

# Function to run backend in subprocess (for env var isolation)
def run_backend(backend_name: str, query: str, top_k: int) -> dict[str, Any]:
    """Run a backend in isolated subprocess."""
    code = f"""
import os
os.environ["PLOT_SEARCH_BACKEND"] = "{backend_name}"

from agent.tools import plot_search

results = plot_search.run("{query.replace('"', '\\\\"')}", top_k={top_k})
import json
print(json.dumps(results, default=str))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd="/Users/alanarazi/tabstar/movibot"
    )

    if result.returncode != 0:
        st.error(f"Error running {backend_name} backend:\n{result.stderr}")
        return []

    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        st.error(f"Failed to parse {backend_name} results")
        return []


def run_full_loop(backend_name: str, query: str) -> dict[str, Any]:
    """Run full ReAct loop in isolated subprocess."""
    code = f"""
import os
os.environ["PLOT_SEARCH_BACKEND"] = "{backend_name}"

from agent import react_loop
result = react_loop.execute("{query.replace('"', '\\\\"')}")
import json
print(json.dumps(result, default=str))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd="/Users/alanarazi/tabstar/movibot",
        timeout=120
    )

    if result.returncode != 0:
        st.error(f"Error running ReAct loop:\n{result.stderr}")
        return {}

    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        st.error("Failed to parse ReAct loop results")
        return {}


# Run the selected backend(s)
if backend_env == "both":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔤 IDF Backend (Keyword Matching)")
        with st.spinner("Running IDF backend..."):
            idf_results = run_backend("idf", user_query, top_k)

        if idf_results:
            idf_df = pd.DataFrame([
                {
                    "Title": r["title"],
                    "Year": r["release_year"],
                    "Score": f"{r['score']:.4f}",
                    "Matched Terms": ", ".join(r.get("matched_terms", [])[:3])
                }
                for r in idf_results
            ])
            st.dataframe(idf_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No results from IDF backend")

    with col2:
        st.subheader("🧠 E5 Backend (Semantic Search)")
        with st.spinner("Running E5 backend..."):
            embed_results = run_backend("embedding", user_query, top_k)

        if embed_results:
            embed_df = pd.DataFrame([
                {
                    "Title": r["title"],
                    "Year": r["release_year"],
                    "Semantic Score": f"{r['score']:.4f}",
                }
                for r in embed_results
            ])
            st.dataframe(embed_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No results from E5 backend")

    # Comparison insights
    st.divider()
    st.subheader("💡 Backend Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **IDF Strengths:**
        - Matches exact keywords in plot text
        - Fast (no model inference)
        - Good for structured queries ("Disney", "1990s")
        """)

    with col2:
        st.markdown("""
        **E5 Strengths:**
        - Finds thematic similarity
        - Handles fuzzy/semantic queries
        - Discovers unexpected matches
        """)

else:
    # Single backend mode
    backend_display = "E5 Embedding (Semantic)" if backend_env == "embedding" else "IDF Keyword Matching"
    st.subheader(f"🎯 {backend_display}")

    with st.spinner(f"Running {backend_display}..."):
        results = run_backend(backend_env, user_query, top_k)

    if results:
        if backend_env == "embedding":
            df = pd.DataFrame([
                {
                    "Title": r["title"],
                    "Year": r["release_year"],
                    "Semantic Score": f"{r['score']:.4f}",
                }
                for r in results
            ])
        else:
            df = pd.DataFrame([
                {
                    "Title": r["title"],
                    "Year": r["release_year"],
                    "Score": f"{r['score']:.4f}",
                    "Matched Terms": ", ".join(r.get("matched_terms", [])[:5])
                }
                for r in results
            ])

        st.dataframe(df, use_container_width=True, hide_index=True)

        # Show full details in expander
        with st.expander("📋 Full Details"):
            for i, r in enumerate(results, 1):
                st.markdown(f"**{i}. {r['title']}** ({r['release_year']})")
                st.caption(f"Score: {r['score']:.4f}")
                if r.get("matched_terms"):
                    st.caption(f"Terms: {', '.join(r['matched_terms'])}")
                st.divider()
    else:
        st.warning("No results found")

# Full ReAct loop section
if show_full_trace:
    st.divider()
    st.subheader("🔄 Full ReAct Loop Trace")
    st.caption(f"Backend: {backend_env}")

    with st.spinner("Running full ReAct loop..."):
        loop_result = run_full_loop(backend_env, user_query)

    if loop_result:
        # Status and response
        col1, col2 = st.columns([1, 3])
        with col1:
            status = loop_result.get("status", "unknown")
            status_color = "🟢" if status == "ok" else "🔴"
            st.metric("Status", f"{status_color} {status}")

        with col2:
            st.text_area(
                "Agent Response",
                value=loop_result.get("response", ""),
                height=150,
                disabled=True
            )

        # Steps trace
        st.subheader("📊 Steps Trace")
        steps = loop_result.get("steps", [])

        for i, step in enumerate(steps, 1):
            module = step.get("module", "Unknown")

            with st.expander(f"Step {i}: {module}"):
                if module == "PlotSearch":
                    response = step.get("response", {})
                    query_used = response.get("query", "N/A")
                    matches = response.get("matches", [])
                    st.write(f"**Query:** {query_used}")
                    st.write(f"**Matches:** {len(matches)} results")

                    if matches:
                        matches_df = pd.DataFrame([
                            {
                                "Title": m["title"],
                                "Year": m["release_year"],
                                "Score": f"{m['score']:.4f}"
                            }
                            for m in matches[:10]
                        ])
                        st.dataframe(matches_df, use_container_width=True, hide_index=True)

                elif module == "CatalogFilter":
                    response = step.get("response", {})
                    candidates = response.get("candidates", [])
                    st.write(f"**Candidates:** {len(candidates)} movies")

                    if candidates:
                        cand_df = pd.DataFrame([
                            {
                                "Title": c["title"],
                                "Year": c["release_year"],
                                "Genre": c.get("genre", "N/A")
                            }
                            for c in candidates[:10]
                        ])
                        st.dataframe(cand_df, use_container_width=True, hide_index=True)

                elif module == "Reasoner":
                    response = step.get("response", {})
                    action = response.get("action", "N/A")
                    reason = response.get("reason", "N/A")
                    st.write(f"**Action:** {action}")
                    st.write(f"**Reason:** {reason}")

                else:
                    st.json(step.get("response", {}))

st.divider()

# Footer
st.markdown("""
---
**About this demo:**
- **E5 Backend:** Semantic search using E5-small-v2 embeddings (384-dim)
- **IDF Backend:** Keyword matching using inverse document frequency scoring
- **Data:** 170 Disney/Pixar movies with plot synopses

See [EMBEDDING_BACKEND.md](EMBEDDING_BACKEND.md) for details.
""")
