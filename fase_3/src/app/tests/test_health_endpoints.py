"""
Tests for the health check endpoint: GET /health

These tests validate the health check functionality used by Docker Compose
and orchestrators to ensure the model is loaded before routing traffic.
"""

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_check_when_model_loaded(self, client: TestClient):
        """Test health check returns 200 when model is loaded."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "status" in data
        assert "model_loaded" in data

        # When model is loaded, status should be "ok"
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_check_response_structure(self, client: TestClient):
        """Test that health check response has correct structure."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Validate field types
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)

    def test_health_check_model_loaded_true(self, client: TestClient):
        """Test that model_loaded is True when inference service is ready."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["model_loaded"] is True

    def test_health_check_response_time(self, client: TestClient):
        """Test that health check responds quickly."""
        import time

        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()

        assert response.status_code == 200
        # Health check should be very fast (< 100ms)
        assert (end_time - start_time) < 0.1

    def test_health_check_no_auth_required(self, client: TestClient):
        """Test that health check doesn't require authentication."""
        response = client.get("/health")

        # Should return 200, not 401 or 403
        assert response.status_code == 200

    def test_health_check_accepts_get_only(self, client: TestClient):
        """Test that health check only accepts GET requests."""
        # POST should not be allowed
        post_response = client.post("/health")
        assert post_response.status_code == 405  # Method Not Allowed

        # PUT should not be allowed
        put_response = client.put("/health")
        assert put_response.status_code == 405

        # DELETE should not be allowed
        delete_response = client.delete("/health")
        assert delete_response.status_code == 405

    def test_health_check_content_type(self, client: TestClient):
        """Test that health check returns JSON content type."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_health_check_consistency(self, client: TestClient):
        """Test that health check returns consistent results across calls."""
        response1 = client.get("/health")
        response2 = client.get("/health")

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Both should return the same status
        assert data1["status"] == data2["status"]
        assert data1["model_loaded"] == data2["model_loaded"]


class TestHealthEndpointEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_health_check_after_startup(self, client: TestClient):
        """Test health check immediately after application startup."""
        # The fixture should have already warmed up the service
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True

    def test_health_check_with_headers(self, client: TestClient):
        """Test health check with various headers."""
        headers = {
            "User-Agent": "TestClient/1.0",
            "Accept": "application/json",
        }
        response = client.get("/health", headers=headers)

        assert response.status_code == 200

    def test_health_check_empty_response_body(self, client: TestClient):
        """Test that health check doesn't return empty body."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Should have content
        assert len(data) > 0

    def test_health_check_no_extra_fields(self, client: TestClient):
        """Test that health check doesn't return unexpected fields."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Should only have expected fields
        expected_fields = {"status", "model_loaded"}
        actual_fields = set(data.keys())
        assert actual_fields == expected_fields


class TestHealthEndpointIntegration:
    """Integration tests for health check with other components."""

    def test_health_check_before_classification(
        self, client: TestClient, sample_clinical_diagnosis
    ):
        """Test that health check can be called before classification."""
        # First check health
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # Then perform classification
        classify_response = client.post("/v1/classify", json={"text": sample_clinical_diagnosis})
        assert classify_response.status_code == 200

    def test_health_check_after_classification(self, client: TestClient, sample_clinical_diagnosis):
        """Test that health check can be called after classification."""
        # First perform classification
        classify_response = client.post("/v1/classify", json={"text": sample_clinical_diagnosis})
        assert classify_response.status_code == 200

        # Then check health
        health_response = client.get("/health")
        assert health_response.status_code == 200

    def test_health_check_during_load(self, client: TestClient):
        """Test health check behavior during normal operation."""
        # Multiple health checks should all succeed
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["model_loaded"] is True
