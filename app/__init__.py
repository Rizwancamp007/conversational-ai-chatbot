import os
import uuid

from flask import Flask, jsonify, render_template, request

from chatbot.dataset import DEFAULT_PATH, load_intents

from .quick_replies import quick_replies
from .rate_limit import RateLimiter

__version__ = "0.1.0"

MAX_MESSAGE_LENGTH = 500
MAX_SESSION_ID_LENGTH = 128


def create_app(config=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or os.urandom(32).hex(),
        CLIENT_URL=os.environ.get("CLIENT_URL", "http://localhost:5000"),
        CONFIDENCE_THRESHOLD=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6")),
        INTENTS_PATH=os.environ.get("INTENTS_PATH", str(DEFAULT_PATH)),
        MAX_MESSAGE_LENGTH=int(os.environ.get("MAX_MESSAGE_LENGTH",
                                              str(MAX_MESSAGE_LENGTH))),
        RATE_LIMIT_PER_MINUTE=int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30")),
        DIALOGUE_MANAGER=None,
        JSON_SORT_KEYS=False,
    )
    if config:
        app.config.update(config)

    limiter = RateLimiter(app.config["RATE_LIMIT_PER_MINUTE"])
    app.extensions["rate_limiter"] = limiter

    def get_manager():
        manager = app.config.get("DIALOGUE_MANAGER")
        if manager is None:
            from chatbot.dialogue.manager import DialogueManager

            manager = DialogueManager()
            app.config["DIALOGUE_MANAGER"] = manager
        return manager

    # -- CORS: only the configured client origin may call the API -----------
    # Same-origin browser calls send no Origin header and are unaffected.
    # The chat UI served here is same-origin, so it needs no CORS header.

    @app.after_request
    def _cors(response):
        origin = request.headers.get("Origin")
        client_url = app.config["CLIENT_URL"].rstrip("/")
        if origin and origin.rstrip("/") == client_url:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        return response

    @app.before_request
    def _preflight():
        if request.method != "OPTIONS" or not request.path.startswith("/api/"):
            return None
        origin = request.headers.get("Origin")
        client_url = app.config["CLIENT_URL"].rstrip("/")
        if origin and origin.rstrip("/") == client_url:
            return ("", 204, {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Max-Age": "86400",
                "Vary": "Origin",
            })
        return "", 204

    def _rate_limited():
        key = request.remote_addr or "unknown"
        if not limiter.allow(key):
            return jsonify(error="too many requests, please wait a moment"), 429
        return None

    # -- pages ---------------------------------------------------------------

    @app.get("/")
    def chat_page():
        return render_template("chat.html")

    # -- health --------------------------------------------------------------

    @app.get("/health")
    def health():
        try:
            intents = len(load_intents(app.config["INTENTS_PATH"])["intents"])
            dataset = "ok"
        except (OSError, ValueError, KeyError, TypeError):
            intents, dataset = 0, "unreadable"
        status = 200 if dataset == "ok" else 503
        return jsonify(status="ok" if status == 200 else "degraded",
                       version=__version__, dataset=dataset, intents=intents), status

    # -- chat API ------------------------------------------------------------

    @app.post("/api/chat")
    def chat():
        limited = _rate_limited()
        if limited is not None:
            return limited

        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="request body must be a JSON object"), 400

        message = body.get("message")
        if not isinstance(message, str) or not message.strip():
            return jsonify(error="message is required"), 400
        message = message.strip()
        if len(message) > app.config["MAX_MESSAGE_LENGTH"]:
            return jsonify(
                error=f"message is too long "
                      f"(max {app.config['MAX_MESSAGE_LENGTH']} characters)"), 400

        session_id = body.get("session_id") or uuid.uuid4().hex
        if not isinstance(session_id, str):
            return jsonify(error="session_id must be a string"), 400
        session_id = session_id.strip() or uuid.uuid4().hex
        if len(session_id) > MAX_SESSION_ID_LENGTH:
            return jsonify(error="session_id is too long"), 400

        result = get_manager().handle(session_id, message)

        from chatbot.nlp.entities import extract_entities

        entities = extract_entities(message)

        return jsonify(
            session_id=result["session_id"],
            reply=result["reply"],
            intent=result["intent"],
            confidence=result["confidence"],
            entities=[{k: e[k] for k in ("type", "value", "text", "start",
                                         "end", "confidence")}
                      for e in entities],
            slots=result["slots"],
            pending_slot=result["pending_slot"],
            handoff_requested=result["handoff_requested"],
            quick_replies=quick_replies(result["pending_slot"]),
        )

    @app.post("/api/session/reset")
    def reset_session():
        limited = _rate_limited()
        if limited is not None:
            return limited

        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="request body must be a JSON object"), 400

        session_id = body.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            return jsonify(error="session_id is required"), 400

        get_manager().reset(session_id.strip())
        return jsonify(session_id=session_id.strip(), reset=True)

    # -- errors, always JSON ---------------------------------------------------

    @app.errorhandler(400)
    def bad_request(_):
        return jsonify(error="bad request"), 400

    @app.errorhandler(404)
    def not_found(_):
        return jsonify(error="not found"), 404

    @app.errorhandler(405)
    def not_allowed(_):
        return jsonify(error="method not allowed"), 405

    @app.errorhandler(500)
    def server_error(_):
        return jsonify(error="something went wrong, please try again"), 500

    return app
