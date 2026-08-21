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


@app.route("/api/budget", methods=["GET"])
def budget():
    """What the project has spent on LLMod.ai, and what it is allowed to spend.

    Read live from the provider rather than typed into TODO.md, where it went
    stale the moment anyone ran a query. Free: it is the proxy's accounting
    endpoint, not a model call.
    """
    from agent import llm_client

    return _cors(jsonify(llm_client.budget()))


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


@app.route("/data/<path:filename>", methods=["GET"])
def data_files(filename):
    # Local dev convenience only, mirroring index() above. In production
    # Vercel's static routing serves public/ directly and never reaches app.py.
    # send_from_directory rejects paths that escape the directory.
    return send_from_directory(os.path.join(BASE_DIR, "public", "data"), filename)


@app.route("/<path:filename>", methods=["GET"])
def public_files(filename):
    """Everything else in public/ -- the hero image, and anything added later.

    Local dev parity only. In production Vercel's static routing already maps
    /(.*) to public/$1 and never reaches Python, so without this the image
    would work on Vercel and 404 locally, which is the wrong way round: it
    would mean shipping an asset that could not be checked before deploying.

    Werkzeug ranks routes with no arguments above routes with them, so this
    cannot shadow the /api/* endpoints defined above.
    """
    return send_from_directory(os.path.join(BASE_DIR, "public"), filename)


@app.route("/api/team_info", methods=["GET"])
def team_info():
    return _cors(jsonify(_load_json(TEAM_INFO_PATH)))


@app.route("/api/agent_info", methods=["GET"])
def agent_info():
    return _cors(jsonify(_load_json(AGENT_INFO_PATH)))


@app.route("/api/model_architecture", methods=["GET"])
def model_architecture():
    return send_file(ARCHITECTURE_PNG_PATH, mimetype="image/png")


def _prompt_token_cost(system_prompt: str, schemas: list) -> dict:
    """Exact cl100k_base counts, or None if the tokenizer is unavailable.

    None rather than an estimate: a wrong number here would be quoted as if
    it were measured.
    """
    try:
        import json as _json

        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        system = len(enc.encode(system_prompt))
        tools_ = len(enc.encode(_json.dumps(schemas)))
    except Exception:
        return {"system_prompt": None, "tool_schemas": None, "per_turn": None}
    return {"system_prompt": system, "tool_schemas": tools_,
            "per_turn": system + tools_, "encoding": "cl100k_base"}


@app.route("/api/prompts", methods=["GET"])
def prompts_endpoint():
    """Every prompt and guardrail the agent actually runs on.

    Served from the modules themselves rather than copied into the page, for the
    same reason rag/DECISIONS.md is: a prompt pasted into HTML is a prompt that
    silently stops matching the one in prompts.py. Anything shown here is the
    live string.
    """
    from agent import answerer as agent_answerer
    from agent import decomposer as agent_decomposer
    from agent import loop as agent_loop
    from agent import tools as agent_tools
    from agent import verifier as agent_verifier
    from rag import screen as rag_screen

    return _cors(jsonify({
        # The three roles, in the order a request meets them. Served from the
        # modules rather than copied into the page, for the reason
        # rag/DECISIONS.md is: a prompt pasted into HTML is a prompt that
        # silently stops matching the one that runs.
        "roles": [
            {
                "name": "QueryDecomposer",
                "does": "reads the request once and returns the plan every later stage runs on",
                "calls": "one per request",
                "prompt": agent_decomposer.DECOMPOSER_PROMPT,
                "output_schema": agent_decomposer.PLAN_SCHEMA,
            },
            {
                "name": "Verifier",
                "does": "one film, every condition, against that film's plot text",
                "calls": f"one per candidate checked, at most {agent_tools.MAX_VERIFICATIONS}",
                "prompt": agent_verifier.VERIFIER_PROMPT,
            },
            {
                "name": "Answerer",
                "does": "writes the reply from what was accepted; cannot reach a film that is not in front of it",
                "calls": "one, plus at most one more if the answer check rejects it",
                "prompt": agent_answerer.ANSWERER_PROMPT,
            },
        ],
        # The deterministic stages between them. Not prompts -- they take
        # arguments from the plan and read local data -- but the descriptions
        # are what a reader needs to follow a trace.
        "stages": [
            {
                "name": agent_tools.TRACE_NAMES[schema["function"]["name"]],
                "function": schema["function"]["name"],
                "description": schema["function"]["description"],
                "parameters": {
                    name: spec.get("description", "")
                    for name, spec in sorted(
                        schema["function"]["parameters"].get("properties", {}).items()
                    )
                },
            }
            for schema in agent_tools.TOOL_SCHEMAS
        ],
        "token_cost": _prompt_token_cost(agent_decomposer.DECOMPOSER_PROMPT,
                                         [agent_decomposer.PLAN_SCHEMA]),
        "guardrails": {
            "MAX_TOTAL_LLM_CALLS": agent_loop.MAX_TOTAL_LLM_CALLS,
            "MAX_VERIFICATIONS": agent_tools.MAX_VERIFICATIONS,
            "MAX_RECOMMENDATIONS": agent_tools.MAX_RECOMMENDATIONS_CEILING,
            "PREVIEW_FILMS": agent_tools.PREVIEW_FILMS,
            "MAX_SEARCH_RESULTS": agent_tools.MAX_SEARCH_RESULTS,
            "MAX_SYNOPSES": agent_tools.MAX_SYNOPSES,
            "MAX_SYNOPSIS_CHARS": agent_tools.MAX_SYNOPSIS_CHARS,
            "MAX_PASSAGE_CHARS": agent_tools.MAX_PASSAGE_CHARS,
            "MAX_FLAGGED_EVIDENCE": agent_tools.MAX_FLAGGED_EVIDENCE,
            "MIN_SCREEN_TOKENS": rag_screen.MIN_SCREEN_TOKENS,
        },
        # No vocabularies and no blacklist any more. Both were fixed lists,
        # and a fixed list only ever fits the requests someone thought of: the
        # planner writes the words it is scanning for and the phrasings that
        # would trip them, per request. The guardrails above are what remain
        # fixed, because they are bounds rather than knowledge.
        "word_lists": (
            "written per request by the planner, not stored. See screen_out's "
            "`words` and `exclude_phrases` in the tool descriptions above."
        ),
    }))


@app.route("/api/execute", methods=["POST", "OPTIONS"])
def execute():
    if request.method == "OPTIONS":
        return _cors(jsonify({}))

    # Both of these are type checks before they are anything else. A JSON body
    # that parses is not a body of the shape we asked for: `[1, 2]` parses to a
    # list, which has no .get, and {"prompt": 123} parses to an int, which has
    # no .strip. Either used to raise here -- above the try below -- so Flask
    # answered with a 500 HTML page instead of the error JSON the spec
    # mandates. Truthiness was never the right test; type is.
    body = request.get_json(silent=True)
    data = body if isinstance(body, dict) else {}
    raw_prompt = data.get("prompt")
    user_prompt = raw_prompt.strip() if isinstance(raw_prompt, str) else ""

    if not user_prompt:
        # Say which of the two it was. "Required" is wrong for {"prompt": 123}:
        # the field was supplied, it was the wrong type, and a caller told to
        # supply a field it already supplied has been sent to look in the wrong
        # place.
        if raw_prompt is None or isinstance(raw_prompt, str):
            message = "The 'prompt' field is required."
        else:
            message = (
                f"The 'prompt' field must be a string, not "
                f"{type(raw_prompt).__name__}."
            )

        # 200, not 400. The spec describes "error" as a response *format* --
        # status, error, response, steps -- and never mentions HTTP codes, so
        # the failure belongs in the body. Every other error this endpoint can
        # produce already returns 200 with that shape: an internal exception, an
        # empty answer from the model, the round bound being exhausted. A
        # missing prompt was the one kind that answered differently, which made
        # a caller's raise_for_status() throw on one error and not the others.
        return _cors(
            jsonify(
                {
                    "status": "error",
                    "error": message,
                    "response": None,
                    "steps": [],
                }
            )
        ), 200

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
