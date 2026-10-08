from pathlib import Path

from fastapi.testclient import TestClient

from grainmaster.web_server import create_app


ROOT = Path(__file__).resolve().parents[1]


def test_public_origin_is_opt_in(monkeypatch, tmp_path):
    public = "https://grainmaster.example.com"
    monkeypatch.delenv("GRAINMASTER_TRUSTED_ORIGINS", raising=False)
    client = TestClient(create_app(ROOT, state_root=tmp_path / "default"))
    response = client.post("/api/process", headers={"Origin": public}, json={})
    assert response.status_code == 403

    monkeypatch.setenv("GRAINMASTER_TRUSTED_ORIGINS", public)
    client = TestClient(create_app(ROOT, state_root=tmp_path / "public"))
    for origin in (public, "http://localhost:8765", "http://100.64.1.2:8765"):
        response = client.post("/api/process", headers={"Origin": origin}, json={})
        assert response.status_code == 422
    for origin in ("https://evil.example.com", public + ".evil.example.com", "http://grainmaster.example.com"):
        response = client.post("/api/process", headers={"Origin": origin}, json={})
        assert response.status_code == 403
