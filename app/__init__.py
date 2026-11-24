import os

from flask import Flask, jsonify

from chatbot.dataset import DEFAULT_PATH, load_intents

__version__ = "0.1.0"


def create_app(config=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or os.urandom(32).hex(),
        CLIENT_URL=os.environ.get("CLIENT_URL", "http://localhost:5000"),
        CONFIDENCE_THRESHOLD=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6")),
        INTENTS_PATH=os.environ.get("INTENTS_PATH", str(DEFAULT_PATH)),
        JSON_SORT_KEYS=False,
    )
    if config:
        app.config.update(config)

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

    @app.errorhandler(404)
    def not_found(_):
        return jsonify(error="not found"), 404

    @app.errorhandler(405)
    def not_allowed(_):
        return jsonify(error="method not allowed"), 405

    return app
