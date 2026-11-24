import pytest

from app import create_app


@pytest.fixture
def client():
    return create_app({"TESTING": True}).test_client()


def test_health_reports_ok_and_dataset(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "ok"
    assert body["dataset"] == "ok"
    assert body["intents"] >= 15


def test_health_degraded_when_dataset_missing(tmp_path):
    app = create_app({"TESTING": True, "INTENTS_PATH": str(tmp_path / "nope.json")})
    res = app.test_client().get("/health")
    assert res.status_code == 503
    assert res.get_json()["status"] == "degraded"


def test_unknown_route_returns_json_404(client):
    res = client.get("/nope")
    assert res.status_code == 404
    assert res.get_json() == {"error": "not found"}


def test_wrong_method_returns_json_405(client):
    res = client.post("/health")
    assert res.status_code == 405
