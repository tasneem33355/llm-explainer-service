import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.explainer import detect_language, build_system_prompt, build_context

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "llm-explainer-service"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "api_key_configured" in response.json()


def test_language_detection():
    assert detect_language("لماذا تم رفض الطلب؟") == "ar"
    assert detect_language("Why was this application rejected?") == "en"
    assert detect_language(None) == "en"


def test_build_context():
    block = {"type": "credit_risk", "data": {"score": 750, "decision": "APPROVE"}}
    ctx = build_context(block["type"], block["data"])
    assert "CREDIT RISK" in ctx
    assert "750" in ctx
