"""
Tests for the model info endpoint: GET /v1/model/info

These tests validate the model information endpoint that exposes
model metadata including version, class labels, and optimization details.
"""

from fastapi.testclient import TestClient


class TestModelInfoEndpoint:
    """Tests for GET /v1/model/info endpoint."""

    def test_model_info_success(self, client: TestClient):
        """Test successful retrieval of model information."""
        response = client.get("/v1/model/info")

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "model_version" in data
        assert "class_labels" in data
        assert "num_classes" in data
        assert "optimization" in data

    def test_model_info_field_types(self, client: TestClient):
        """Test that model info fields have correct types."""
        response = client.get("/v1/model/info")

        assert response.status_code == 200
        data = response.json()

        # Validate field types
        assert isinstance(data["model_version"], str)
        assert isinstance(data["class_labels"], list)
        assert isinstance(data["num_classes"], int)
        assert isinstance(data["optimization"], str)

    def test_model_info_class_labels_content(self, client: TestClient):
        """Test that class labels contain expected medical categories."""
        response = client.get("/v1/model/info")

        assert response.status_code == 200
        data = response.json()

        # Validate that class labels match expected categories
        expected_labels = [
            "Neoplasms",
            "Digestive system diseases",
            "Nervous system diseases",
            "Cardiovascular diseases",
            "General pathological conditions",
        ]

        assert set(data["class_labels"]) == set(expected_labels)

    def test_model_info_num_classes_matches_labels(self, client: TestClient):
        """Test that num_classes matches the length of class_labels."""
        response = client.get("/v1/model/info")

        assert response.status_code == 200
        data = response.json()

        assert data["num_classes"] == len(data["class_labels"])

    def test_model_info_expected_num_classes(self, client: TestClient):
        """Test that num_classes equals the expected value (5)."""
        response = client.get("/v1/model/info")

        assert response.status_code == 200
        data = response.json()

        # The model should have 5 classes based on the medical abstract dataset
        assert data["num_classes"] == 5

    def test_model_info_optimization_field(self, client: TestClient):
        """Test that optimization field indicates ONNX."""
        response = client.get("/v1/model/info")

        assert response.status_code == 200
        data = response.json()

        # Should indicate ONNX optimization
        assert data["optimization"] == "onnx"

    def test_model_info_version_format(self, client: TestClient):
        """Test that model version follows expected format."""
        response = client.get("/v1/model/info")

        assert response.status_code == 200
        data = response.json()

        # Version should be a non-empty string
        assert isinstance(data["model_version"], str)
        assert len(data["model_version"]) > 0

    def test_model_info_response_time(self, client: TestClient):
        """Test that model info responds quickly."""
        import time

        start_time = time.time()
        response = client.get("/v1/model/info")
        end_time = time.time()

        assert response.status_code == 200
        # Model info should be very fast (< 50ms)
        assert (end_time - start_time) < 0.05

    def test_model_info_no_auth_required(self, client: TestClient):
        """Test that model info doesn't require authentication."""
        response = client.get("/v1/model/info")

        # Should return 200, not 401 or 403
        assert response.status_code == 200

    def test_model_info_accepts_get_only(self, client: TestClient):
        """Test that model info only accepts GET requests."""
        # POST should not be allowed
        post_response = client.post("/v1/model/info")
        assert post_response.status_code == 405  # Method Not Allowed

        # PUT should not be allowed
        put_response = client.put("/v1/model/info")
        assert put_response.status_code == 405

        # DELETE should not be allowed
        delete_response = client.delete("/v1/model/info")
        assert delete_response.status_code == 405

    def test_model_info_content_type(self, client: TestClient):
        """Test that model info returns JSON content type."""
        response = client.get("/v1/model/info")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"


class TestModelInfoConsistency:
    """Tests for model info consistency and reliability."""

    def test_model_info_consistency_across_calls(self, client: TestClient):
        """Test that model info returns consistent data across multiple calls."""
        response1 = client.get("/v1/model/info")
        response2 = client.get("/v1/model/info")

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # All fields should be consistent
        assert data1["model_version"] == data2["model_version"]
        assert data1["class_labels"] == data2["class_labels"]
        assert data1["num_classes"] == data2["num_classes"]
        assert data1["optimization"] == data2["optimization"]

    def test_model_info_no_extra_fields(self, client: TestClient):
        """Test that model info doesn't return unexpected fields."""
        response = client.get("/v1/model/info")

        assert response.status_code == 200
        data = response.json()

        # Should only have expected fields
        expected_fields = {"model_version", "class_labels", "num_classes", "optimization"}
        actual_fields = set(data.keys())
        assert actual_fields == expected_fields

    def test_model_info_class_labels_order(self, client: TestClient):
        """Test that class labels maintain consistent order."""
        response1 = client.get("/v1/model/info")
        response2 = client.get("/v1/model/info")

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Order should be consistent
        assert data1["class_labels"] == data2["class_labels"]


class TestModelInfoIntegration:
    """Integration tests for model info with other components."""

    def test_model_info_before_classification(self, client: TestClient, sample_clinical_diagnosis):
        """Test that model info can be called before classification."""
        # First get model info
        info_response = client.get("/v1/model/info")
        assert info_response.status_code == 200
        info_data = info_response.json()

        # Then perform classification
        classify_response = client.post("/v1/classify", json={"text": sample_clinical_diagnosis})
        assert classify_response.status_code == 200
        classify_data = classify_response.json()

        # Model version should match
        assert info_data["model_version"] == classify_data["model_version"]

    def test_model_info_after_classification(self, client: TestClient, sample_clinical_diagnosis):
        """Test that model info can be called after classification."""
        # First perform classification
        classify_response = client.post("/v1/classify", json={"text": sample_clinical_diagnosis})
        assert classify_response.status_code == 200
        classify_data = classify_response.json()

        # Then get model info
        info_response = client.get("/v1/model/info")
        assert info_response.status_code == 200
        info_data = info_response.json()

        # Model version should match
        assert info_data["model_version"] == classify_data["model_version"]

    def test_model_info_batch_classification_consistency(
        self, client: TestClient, sample_clinical_diagnoses
    ):
        """Test that model info is consistent with batch classification."""
        # Get model info
        info_response = client.get("/v1/model/info")
        assert info_response.status_code == 200
        info_data = info_response.json()

        # Perform batch classification
        batch_response = client.post(
            "/v1/classify/batch", json={"texts": sample_clinical_diagnoses}
        )
        assert batch_response.status_code == 200
        batch_data = batch_response.json()

        # Model version should match
        assert info_data["model_version"] == batch_data["model_version"]

        # All predicted labels should be from the class_labels
        for result in batch_data["results"]:
            assert result["label"] in info_data["class_labels"]

    def test_model_info_labels_match_classification_outputs(
        self, client: TestClient, clinical_diagnosis_by_category
    ):
        """Test that model info labels match actual classification outputs."""
        # Get model info
        info_response = client.get("/v1/model/info")
        assert info_response.status_code == 200
        info_data = info_response.json()

        # Classify one sample from each category
        all_diagnoses = []
        for category_diagnoses in clinical_diagnosis_by_category.values():
            all_diagnoses.extend(category_diagnoses[:1])  # Take one from each

        batch_response = client.post("/v1/classify/batch", json={"texts": all_diagnoses})
        assert batch_response.status_code == 200
        batch_data = batch_response.json()

        # All predicted labels should be in the model info class labels
        valid_labels = set(info_data["class_labels"])
        for result in batch_data["results"]:
            assert result["label"] in valid_labels

    def test_model_info_with_health_check(self, client: TestClient):
        """Test that model info and health check can both be called successfully."""
        # Get model info
        info_response = client.get("/v1/model/info")
        assert info_response.status_code == 200

        # Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # Both should succeed independently
        info_data = info_response.json()
        health_data = health_response.json()

        assert info_data["model_version"] is not None
        assert health_data["model_loaded"] is True


class TestModelInfoEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_model_info_with_query_params(self, client: TestClient):
        """Test that model info ignores query parameters."""
        response = client.get("/v1/model/info?format=json&verbose=true")

        # Should still return 200
        assert response.status_code == 200

    def test_model_info_with_headers(self, client: TestClient):
        """Test model info with various headers."""
        headers = {
            "User-Agent": "TestClient/1.0",
            "Accept": "application/json",
        }
        response = client.get("/v1/model/info", headers=headers)

        assert response.status_code == 200

    def test_model_info_empty_response_body(self, client: TestClient):
        """Test that model info doesn't return empty body."""
        response = client.get("/v1/model/info")

        assert response.status_code == 200
        data = response.json()

        # Should have content
        assert len(data) > 0

    def test_model_info_class_labels_not_empty(self, client: TestClient):
        """Test that class labels list is not empty."""
        response = client.get("/v1/model/info")

        assert response.status_code == 200
        data = response.json()

        assert len(data["class_labels"]) > 0

    def test_model_info_num_classes_positive(self, client: TestClient):
        """Test that num_classes is a positive integer."""
        response = client.get("/v1/model/info")

        assert response.status_code == 200
        data = response.json()

        assert data["num_classes"] > 0
