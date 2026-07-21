"""Phase 0 API tests: health check + that the info endpoint carries the disclaimer."""
from fastapi.testclient import TestClient

from biosignal_api.main import DISCLAIMER, app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root_carries_disclaimer():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["disclaimer"] == DISCLAIMER
    assert "NOT an approved medical" in body["disclaimer"]
    assert body["dataset"].startswith("PPG-DaLiA")
