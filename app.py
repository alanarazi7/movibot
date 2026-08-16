import hmac
import json
import os

from flask import Flask, jsonify, request, send_file, send_from_directory

# Local development reads credentials from .env; on Vercel they come from the
# project's environment variables and no .env file exists. load_dotenv() is a
# no-op in that case, so the same entry point serves both.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional in production
    pass

from agent import loop

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEAM_INFO_PATH = os.path.join(BASE_DIR, "team_info.json")
AGENT_INFO_PATH = os.path.join(BASE_DIR, "agent_info.json")
ARCHITECTURE_PNG_PATH = os.path.join(BASE_DIR, "assets", "architecture.png")
GUI_PATH = os.path.join(BASE_DIR, "public", "index.html")
TODO_PATH = os.path.join(BASE_DIR, "TODO.md")
RAG_DECISIONS_PATH = os.path.join(BASE_DIR, "rag", "DECISIONS.md")

app = Flask(__name__)


def _cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.route("/", methods=["GET"])
def index():
    # Local dev convenience only. In production, Vercel's static routing
    # (see vercel.json) serves public/ directly and never hits app.py for "/".
    return send_file(GUI_PATH)


@app.route("/api/status", methods=["GET"])
def status():
    # Served from TODO.md rather than duplicated into the GUI, so the checklist
    # the team edits is the one the page shows. team_info.json proves non-.py
    # files ship in the serverless bundle, so this works in production too.
    try:
        with open(TODO_PATH, "r", encoding="utf-8") as f:
            return _cors(jsonify({"markdown": f.read()}))
    except FileNotFoundError:
        return _cors(jsonify({"markdown": None, "error": "TODO.md not bundled."})), 404


@app.route("/api/rag/decisions", methods=["GET"])
def rag_decisions():
    with open(RAG_DECISIONS_PATH, "r", encoding="utf-8") as f:
        return _cors(jsonify({"markdown": f.read()}))


@app.route("/api/rag/info", methods=["GET"])
def rag_info():
    """Corpora, parameters, and whether the index is currently usable."""
    from rag import config as ragcfg, corpora, store

    try:
        index = store.coverage()
        error = None
    except Exception as exc:
        index, error = None, str(exc)

    return _cors(jsonify({
        "parameters": ragcfg.as_dict(),
        "corpora": corpora.catalogue(),
        "default_sources": corpora.DEFAULT_SOURCES,
        "index": index,
        "index_error": error,
        "ingest_enabled": _ingest_enabled(),
        "index_writable": _index_writable(),
    }))


@app.route("/api/rag/search", methods=["POST", "OPTIONS"])
def rag_search():
    """Semantic search, optionally restricted to particular corpora.

    Costs one embedding call, about two millionths of a dollar.
    """
    if request.method == "OPTIONS":
        return _cors(jsonify({}))

    from rag import corpora, store

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return _cors(jsonify({"error": "The 'query' field is required.",
                              "results": []})), 400

    requested = data.get("sources")
    sources = corpora.resolve(requested) if requested else None
    top_k = max(1, min(int(data.get("top_k") or 10), 50))

    try:
        passages = store.search_passages(query, top_k=top_k, sources=sources)
    except Exception as exc:
        return _cors(jsonify({"error": str(exc), "results": []})), 200

    return _cors(jsonify({
        "error": None,
        "query": query,
        "sources": sources or "all",
        "results": passages,
    }))


def _ingest_enabled() -> bool:
    """Ingest spends money, so it is off unless deliberately switched on."""
    return os.environ.get("MOVIBOT_ALLOW_INGEST", "").strip().lower() in ("1", "true", "yes")


def _index_writable() -> bool:
    """Can the passage index actually be written here?

    Probed rather than inferred from an env var. Vercel's serverless
    filesystem is read-only outside /tmp, so an ingest there would embed the
    whole corpus, spend the money, and only then fail on write -- the worst
    possible ordering. Pinecone is unaffected: it writes over the network.
    """
    from rag.config import DATA_READY

    return os.path.isdir(DATA_READY) and os.access(DATA_READY, os.W_OK)


@app.route("/api/rag/ingest", methods=["POST", "OPTIONS"])
def rag_ingest():
    if request.method == "OPTIONS":
        return _cors(jsonify({}))

    from rag import corpora

    data = request.get_json(silent=True) or {}
    sources = corpora.resolve(data.get("sources")) or corpora.DEFAULT_SOURCES
    to_pinecone = bool(data.get("pinecone"))
    # Debug is the default: an ingest triggered without saying which kind
    # should be the cheap one. The full run has to be asked for.
    debug = bool(data.get("debug", True))
    dry_run = bool(data.get("dry_run", True))

    # The gate exists to stop a stranger spending the budget, so it applies to
    # spending only. Estimating costs nothing and stays available everywhere --
    # otherwise the free button on the public page would 403.
    if not dry_run:
        if not _ingest_enabled():
            return _cors(jsonify({
                "error": "Ingest is disabled here. It embeds the corpus and costs "
                         "money, so it is not exposed by default. Set "
                         "MOVIBOT_ALLOW_INGEST=1 to enable, or run it from a shell: "
                         "python -m rag.ingest --sources all --pinecone",
                "started": False,
            })), 403

        # Anyone may trigger this, so it must be impossible to spend money on a
        # run that cannot succeed. Every destination is checked BEFORE
        # embedding: failing afterwards would mean paying for vectors that are
        # then thrown away, which is exactly how the budget got charged once
        # already.
        pinecone_key = os.environ.get("PINECONE_API_KEY", "")
        pinecone_ready = bool(pinecone_key) and "your-" not in pinecone_key
        destinations = []
        if _index_writable():
            destinations.append("matrix")
        if to_pinecone and pinecone_ready:
            destinations.append("pinecone")

        if not destinations:
            return _cors(jsonify({
                "error": "Nowhere to put the vectors, so nothing was embedded and "
                         "nothing was spent. This filesystem is read-only (as on "
                         "Vercel) so the committed matrix cannot be rebuilt here"
                         + ("" if pinecone_ready else ", and PINECONE_API_KEY is not set")
                         + ". Run ingest locally and commit the index instead.",
                "started": False,
            })), 409

    try:
        from rag import ingest as rag_ingest_mod
        chunks = rag_ingest_mod.build_chunks(sources, debug=debug)
        if dry_run:
            tokens = int(chunks.tokens.sum())
            return _cors(jsonify({
                "started": False, "dry_run": True, "error": None,
                "passages": int(len(chunks)),
                "films": int(chunks.movie_id.nunique()),
                "tokens": tokens,
                "debug": debug,
                "estimated_cost_usd": round(tokens / 1e6 * 0.02, 4),
            }))

        from rag import embed as rag_embed
        vectors, stats = rag_embed.embed_texts_cached(chunks.embedding_text.tolist())
        if "matrix" in destinations:
            rag_ingest_mod.write_index(chunks, vectors, sources, debug=debug)
        if "pinecone" in destinations:
            rag_ingest_mod.upsert_pinecone(chunks, vectors)
        return _cors(jsonify({
            "started": True, "dry_run": False, "error": None, "debug": debug,
            "passages": int(len(chunks)), "destinations": destinations,
            "reused": stats["reused"], "embedded": stats["embedded"],
        }))
    except Exception as exc:
        return _cors(jsonify({"started": False, "error": f"{type(exc).__name__}: {exc}"})), 200


@app.route("/data/<path:filename>", methods=["GET"])
def data_files(filename):
    # Local dev convenience only, mirroring index() above. In production
    # Vercel's static routing serves public/ directly and never reaches app.py.
    # send_from_directory rejects paths that escape the directory.
    return send_from_directory(os.path.join(BASE_DIR, "public", "data"), filename)


@app.route("/api/team_info", methods=["GET"])
def team_info():
    return _cors(jsonify(_load_json(TEAM_INFO_PATH)))


@app.route("/api/agent_info", methods=["GET"])
def agent_info():
    return _cors(jsonify(_load_json(AGENT_INFO_PATH)))


@app.route("/api/model_architecture", methods=["GET"])
def model_architecture():
    return send_file(ARCHITECTURE_PNG_PATH, mimetype="image/png")


@app.route("/api/execute", methods=["POST", "OPTIONS"])
def execute():
    if request.method == "OPTIONS":
        return _cors(jsonify({}))

    data = request.get_json(silent=True) or {}
    user_prompt = (data.get("prompt") or "").strip()

    if not user_prompt:
        return _cors(
            jsonify(
                {
                    "status": "error",
                    "error": "The 'prompt' field is required.",
                    "response": None,
                    "steps": [],
                }
            )
        ), 400

    try:
        result = loop.execute(user_prompt)
    except Exception as exc:
        # Defense-in-depth: loop.execute already catches internally
        return _cors(
            jsonify(
                {
                    "status": "error",
                    "error": f"Internal agent error: {exc}",
                    "response": None,
                    "steps": [],
                }
            )
        ), 200

    return _cors(jsonify(result)), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
