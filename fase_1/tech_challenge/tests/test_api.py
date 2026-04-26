from fastapi.testclient import TestClient

from src.api.main import app

# Initialize the TestClient with the FastAPI app
client = TestClient(app)


def test_health_check():
    """
    Validates the GET /api/health endpoint.
    Ref: README.md - API Endpoints Section 1
    """
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert data["model_loaded"] is True


def test_model_info():
    """
    Validates the GET /api/model-info endpoint.
    Ref: README.md - API Endpoints Section 2
    """
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "features_used" in data
    assert isinstance(data["features_used"], list)


def test_predict_high_risk():
    """
    Validates the POST /api/predict endpoint with a high-risk scenario.
    Ref: README.md - Example Use Cases (Alto Risco)
    """
    high_risk_payload = {
        "features": {
            "gender": "Male",
            "senior_citizen": "No",
            "partner": "No",
            "dependents": "No",
            "tenure_months": 2,
            "phone_service": "Yes",
            "multiple_lines": "No",
            "internet_service": "DSL",
            "online_security": "No",
            "online_backup": "No",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "contract": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Mailed check",
            "monthly_charges": 53.85,
            "total_charges": 108.15,
        }
    }
    response = client.post("/api/predict", json=high_risk_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    # High risk should ideally return 1, but we check if probability is > 0.5
    assert data["prediction"] == 1
    assert data["probability"] > 0.5


def test_predict_batch(client):
    """Valida a predição em lote com múltiplos registros em snake_case."""
    batch_payload = {
        "samples": [
            {
                "gender": "Female",
                "senior_citizen": "No",
                "partner": "Yes",
                "dependents": "No",
                "tenure_months": 12,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "Yes",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "No",
                "streaming_movies": "No",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 70.0,
                "total_charges": 840.0,
            },
            {
                "gender": "Male",
                "senior_citizen": "No",
                "partner": "No",
                "dependents": "No",
                "tenure_months": 24,
                "phone_service": "Yes",
                "multiple_lines": "Yes",
                "internet_service": "DSL",
                "online_security": "Yes",
                "online_backup": "No",
                "device_protection": "Yes",
                "tech_support": "Yes",
                "streaming_tv": "No",
                "streaming_movies": "No",
                "contract": "One year",
                "paperless_billing": "No",
                "payment_method": "Mailed check",
                "monthly_charges": 60.0,
                "total_charges": 1440.0,
            },
        ],
        "return_probabilities": True,
    }

    response = client.post("/api/predict-batch", json=batch_payload)

    assert response.status_code == 200
    data = response.json()

    # Now this will pass because len(data["predictions"]) will be 2
    print(data)  # Debug print to see the actual response
    assert len(data["predictions"]) == 2
    assert "probabilities" in data
    assert data["batch_size"] == 2


def test_predict_invalid_data(client):
    # Envia dados que violam o esquema (faltando campos obrigatórios)
    payload = {"features": {"monthly_charges": "not_a_number"}}
    response = client.post("/api/predict", json=payload)

    # 1. Com o novo esquema, o FastAPI retorna 422 automaticamente
    assert response.status_code == 422

    # 2. Verifique se a mensagem de erro menciona os campos ausentes
    errors = response.json()["detail"]
    assert any(err["loc"] == ["body", "features", "gender"] for err in errors)
    assert any("missing" in err["type"] for err in errors)
