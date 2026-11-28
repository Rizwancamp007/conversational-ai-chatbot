import pytest

from app import create_app


class StubManager:
    """Stand-in for DialogueManager: answers without loading the model."""

    def __init__(self):
        self.calls = []
        self.resets = []
        self.next_result = {
            "session_id": "stub-session",
            "reply": "Which course do you want the fee for?",
            "intent": "fees",
            "confidence": 0.9,
            "slots": {},
            "pending_slot": "course",
            "handoff_requested": False,
        }

    def handle(self, session_id, text):
        self.calls.append((session_id, text))
        result = dict(self.next_result)
        result["session_id"] = session_id
        return result

    def reset(self, session_id):
        self.resets.append(session_id)


@pytest.fixture
def stub():
    return StubManager()


@pytest.fixture
def client(stub):
    app = create_app({
        "TESTING": True,
        "DIALOGUE_MANAGER": stub,
        "RATE_LIMIT_PER_MINUTE": 0,  # off unless a test turns it on
        "CLIENT_URL": "https://chat.example.com",
    })
    return app.test_client()


def chat(client, **body):
    return client.post("/api/chat", json=body)


def test_chat_returns_reply_intent_entities_and_chips(client, stub):
    res = chat(client, session_id="s1", message="FSc fees")
    assert res.status_code == 200
    body = res.get_json()
    assert body["session_id"] == "s1"
    assert body["reply"] == "Which course do you want the fee for?"
    assert body["intent"] == "fees"
    assert body["confidence"] == 0.9
    assert body["pending_slot"] == "course"
    assert body["handoff_requested"] is False
    assert body["slots"] == {}
    assert body["quick_replies"] == ["FSc", "ICS", "FA", "ICOM"]
    assert any(e["type"] == "course" and e["value"] == "fsc"
               for e in body["entities"])
    assert stub.calls == [("s1", "FSc fees")]


def test_chat_generates_session_id_when_missing(client):
    res = chat(client, message="hello")
    assert res.status_code == 200
    assert res.get_json()["session_id"]


def test_chat_topic_chips_when_nothing_pending(client, stub):
    stub.next_result["pending_slot"] = ""
    res = chat(client, session_id="s1", message="hello")
    assert res.get_json()["quick_replies"] == [
        "Admissions", "Fees", "Courses", "Timetable", "Tests", "Results"]


def test_chat_rejects_missing_message(client):
    res = chat(client, session_id="s1")
    assert res.status_code == 400
    assert res.get_json() == {"error": "message is required"}


def test_chat_rejects_empty_and_non_string_message(client):
    assert chat(client, message="   ").status_code == 400
    assert chat(client, message=42).status_code == 400


def test_chat_rejects_overlong_message(client):
    res = chat(client, message="x" * 501)
    assert res.status_code == 400
    assert "too long" in res.get_json()["error"]


def test_chat_rejects_non_json_body(client):
    res = client.post("/api/chat", data="not json",
                      content_type="application/json")
    assert res.status_code == 400
    assert res.get_json() == {"error": "request body must be a JSON object"}


def test_chat_rejects_non_string_session_id(client):
    res = chat(client, session_id=123, message="hello")
    assert res.status_code == 400


def test_reset_clears_session(client, stub):
    res = client.post("/api/session/reset", json={"session_id": "s1"})
    assert res.status_code == 200
    assert res.get_json() == {"session_id": "s1", "reset": True}
    assert stub.resets == ["s1"]


def test_reset_rejects_missing_session_id(client):
    res = client.post("/api/session/reset", json={})
    assert res.status_code == 400
    assert res.get_json() == {"error": "session_id is required"}


def test_rate_limit_blocks_after_limit(stub):
    app = create_app({
        "TESTING": True,
        "DIALOGUE_MANAGER": stub,
        "RATE_LIMIT_PER_MINUTE": 3,
    })
    client = app.test_client()
    for _ in range(3):
        assert chat(client, message="hello").status_code == 200
    res = chat(client, message="hello")
    assert res.status_code == 429
    assert res.get_json() == {
        "error": "too many requests, please wait a moment"}


def test_cors_allows_configured_client_origin(client):
    res = chat(client, message="hello")
    assert "Access-Control-Allow-Origin" not in res.headers
    res = client.post("/api/chat", json={"message": "hello"},
                      headers={"Origin": "https://chat.example.com"})
    assert res.headers["Access-Control-Allow-Origin"] == \
        "https://chat.example.com"
    res = client.post("/api/chat", json={"message": "hello"},
                      headers={"Origin": "https://someone-else.example"})
    assert "Access-Control-Allow-Origin" not in res.headers


def test_preflight_for_configured_origin(client):
    res = client.options(
        "/api/chat", headers={
            "Origin": "https://chat.example.com",
            "Access-Control-Request-Method": "POST",
        })
    assert res.status_code == 204
    assert res.headers["Access-Control-Allow-Origin"] == \
        "https://chat.example.com"


def test_chat_page_serves_ui(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Academy Assistant" in html
    assert 'id="messages"' in html
    assert 'id="message-input"' in html
    assert "New chat" in html
    assert "chat.css" in html
    assert "chat.js" in html


def test_chat_static_assets_serve(client):
    assert client.get("/static/chat.css").status_code == 200
    js = client.get("/static/chat.js")
    assert js.status_code == 200
    assert "/api/chat" in js.get_data(as_text=True)
    assert "/api/session/reset" in js.get_data(as_text=True)
