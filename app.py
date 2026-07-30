import json
import os

from flask import Flask, jsonify, request, send_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEAM_INFO_PATH = os.path.join(BASE_DIR, "team_info.json")
AGENT_INFO_PATH = os.path.join(BASE_DIR, "agent_info.json")
ARCHITECTURE_PNG_PATH = os.path.join(BASE_DIR, "assets", "architecture.png")
GUI_PATH = os.path.join(BASE_DIR, "public", "index.html")

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

    # SKELETON STUB: no LLMod.ai / Pinecone / Supabase calls yet. This
    # returns a hardcoded response in the exact required shape so the API
    # contract and GUI can be reviewed before the real ReAct loop
    # (agent/react_loop.py) is wired in.
    stub_response = (
        "STUB RESPONSE - the real agent is not wired up yet in this skeleton pass. "
        f"You asked: \"{user_prompt}\". Once the ReAct loop (agent/react_loop.py) is "
        "connected, this will run CatalogFilter / PlotSearch / SceneSearch / "
        "ExternalContext against real data and return a verified answer."
    )
    stub_steps = [
        {
            "module": "Reasoner",
            "prompt": {
                "system_prompt": "STUB - You are MoviBot's planner. Decide the next action.",
                "user_prompt": f"STUB - User request: {user_prompt}",
            },
            "response": {
                "action": "CatalogFilter",
                "reason": "STUB - placeholder plan, no real reasoning performed.",
            },
        },
        {
            "module": "Synthesizer",
            "prompt": {
                "system_prompt": "STUB - Compose the final answer from gathered evidence.",
                "user_prompt": "STUB - no real evidence gathered in this skeleton pass.",
            },
            "response": {"answer": stub_response},
        },
    ]

    return _cors(
        jsonify(
            {
                "status": "ok",
                "error": None,
                "response": stub_response,
                "steps": stub_steps,
            }
        )
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
