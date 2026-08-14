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
