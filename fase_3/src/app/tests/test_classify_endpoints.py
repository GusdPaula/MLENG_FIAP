"""
Tests for the classification endpoints: POST /v1/classify and POST /v1/classify/batch

These tests validate the API endpoints using clinical diagnosis data similar to
the treino_modelo dataset structure, ensuring the API correctly processes medical
abstracts and returns valid classification results.
"""

from fastapi.testclient import TestClient


class TestSingleClassifyEndpoint:
    """Tests for POST /v1/classify (single item classification)."""

    def test_single_classify_success(self, client: TestClient, sample_clinical_diagnosis):
        """Test successful classification of a single clinical diagnosis."""
        response = client.post("/v1/classify", json={"text": sample_clinical_diagnosis})

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "label" in data
        assert "confidence" in data
        assert "scores" in data
        assert "model_version" in data
        assert "inference_ms" in data

        # Validate data types and ranges
        assert isinstance(data["label"], str)
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["scores"], dict)
        assert isinstance(data["model_version"], str)
        assert isinstance(data["inference_ms"], float)
        assert data["inference_ms"] > 0

        # Validate scores match confidence
        assert data["scores"][data["label"]] == data["confidence"]

    def test_single_classify_neoplasms(self, client: TestClient, clinical_diagnosis_by_category):
        """Test classification of neoplasms category."""
        diagnosis = clinical_diagnosis_by_category["neoplasms"][0]
        response = client.post("/v1/classify", json={"text": diagnosis})

        assert response.status_code == 200
        data = response.json()
        assert data["label"] in [
            "Neoplasms",
            "Digestive system diseases",
            "Nervous system diseases",
            "Cardiovascular diseases",
            "General pathological conditions",
        ]

    def test_single_classify_digestive_diseases(
        self, client: TestClient, clinical_diagnosis_by_category
    ):
        """Test classification of digestive system diseases."""
        diagnosis = clinical_diagnosis_by_category["digestive_system_diseases"][0]
        response = client.post("/v1/classify", json={"text": diagnosis})

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["label"], str)

    def test_single_classify_nervous_diseases(
        self, client: TestClient, clinical_diagnosis_by_category
    ):
        """Test classification of nervous system diseases."""
        diagnosis = clinical_diagnosis_by_category["nervous_system_diseases"][0]
        response = client.post("/v1/classify", json={"text": diagnosis})

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["label"], str)

    def test_single_classify_cardiovascular_diseases(
        self, client: TestClient, clinical_diagnosis_by_category
    ):
        """Test classification of cardiovascular diseases."""
        diagnosis = clinical_diagnosis_by_category["cardiovascular_diseases"][0]
        response = client.post("/v1/classify", json={"text": diagnosis})

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["label"], str)

    def test_single_classify_general_conditions(
        self, client: TestClient, clinical_diagnosis_by_category
    ):
        """Test classification of general pathological conditions."""
        diagnosis = clinical_diagnosis_by_category["general_pathological_conditions"][0]
        response = client.post("/v1/classify", json={"text": diagnosis})

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["label"], str)

    def test_single_classify_empty_text(self, client: TestClient):
        """Test that empty text is rejected."""
        response = client.post("/v1/classify", json={"text": ""})

        # Should return validation error
        assert response.status_code == 422

    def test_single_classify_whitespace_only(self, client: TestClient, whitespace_only_diagnosis):
        """Test that whitespace-only text is rejected."""
        response = client.post("/v1/classify", json={"text": whitespace_only_diagnosis})

        # Should return validation error
        assert response.status_code == 422

    def test_single_classify_missing_text_field(self, client: TestClient):
        """Test that missing text field is rejected."""
        response = client.post("/v1/classify", json={})

        # Should return validation error
        assert response.status_code == 422

    def test_single_classify_invalid_content_type(self, client: TestClient):
        """Test that invalid content type is rejected."""
        response = client.post(
            "/v1/classify",
            data="text=plain text",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        # Should return validation error
        assert response.status_code == 422


class TestBatchClassifyEndpoint:
    """Tests for POST /v1/classify/batch (batch classification)."""

    def test_batch_classify_success(self, client: TestClient, sample_clinical_diagnoses):
        """Test successful batch classification of clinical diagnoses."""
        response = client.post("/v1/classify/batch", json={"texts": sample_clinical_diagnoses})

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "results" in data
        assert "model_version" in data
        assert "batch_size" in data
        assert "inference_ms" in data

        # Validate batch metadata
        assert data["batch_size"] == len(sample_clinical_diagnoses)
        assert isinstance(data["model_version"], str)
        assert isinstance(data["inference_ms"], float)
        assert data["inference_ms"] > 0

        # Validate results array
        assert len(data["results"]) == len(sample_clinical_diagnoses)

        # Validate each result
        for result in data["results"]:
            assert "label" in result
            assert "confidence" in result
            assert "scores" in result
            assert isinstance(result["label"], str)
            assert isinstance(result["confidence"], float)
            assert 0.0 <= result["confidence"] <= 1.0
            assert isinstance(result["scores"], dict)

    def test_batch_classify_all_categories(
        self, client: TestClient, clinical_diagnosis_by_category
    ):
        """Test batch classification with samples from all categories."""
        # Combine all categories
        all_diagnoses = []
        for category_diagnoses in clinical_diagnosis_by_category.values():
            all_diagnoses.extend(category_diagnoses)

        response = client.post("/v1/classify/batch", json={"texts": all_diagnoses})

        assert response.status_code == 200
        data = response.json()

        assert data["batch_size"] == len(all_diagnoses)
        assert len(data["results"]) == len(all_diagnoses)

    def test_batch_classify_single_item(self, client: TestClient, sample_clinical_diagnosis):
        """Test batch classification with a single item."""
        response = client.post("/v1/classify/batch", json={"texts": [sample_clinical_diagnosis]})

        assert response.status_code == 200
        data = response.json()

        assert data["batch_size"] == 1
        assert len(data["results"]) == 1

    def test_batch_classify_large_batch(self, client: TestClient, sample_clinical_diagnoses_batch):
        """Test batch classification with a larger batch."""
        response = client.post(
            "/v1/classify/batch", json={"texts": sample_clinical_diagnoses_batch}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["batch_size"] == len(sample_clinical_diagnoses_batch)
        assert len(data["results"]) == len(sample_clinical_diagnoses_batch)

    def test_batch_classify_empty_list(self, client: TestClient):
        """Test that empty list is rejected."""
        response = client.post("/v1/classify/batch", json={"texts": []})

        # Should return validation error
        assert response.status_code == 422

    def test_batch_classify_with_empty_string(self, client: TestClient, sample_clinical_diagnoses):
        """Test that batch with empty string is rejected."""
        texts_with_empty = sample_clinical_diagnoses + [""]
        response = client.post("/v1/classify/batch", json={"texts": texts_with_empty})

        # Should return validation error
        assert response.status_code == 422

    def test_batch_classify_with_whitespace(
        self, client: TestClient, sample_clinical_diagnoses, whitespace_only_diagnosis
    ):
        """Test that batch with whitespace-only entry is rejected."""
        texts_with_whitespace = sample_clinical_diagnoses + [whitespace_only_diagnosis]
        response = client.post("/v1/classify/batch", json={"texts": texts_with_whitespace})

        # Should return validation error
        assert response.status_code == 422

    def test_batch_classify_missing_texts_field(self, client: TestClient):
        """Test that missing texts field is rejected."""
        response = client.post("/v1/classify/batch", json={})

        # Should return validation error
        assert response.status_code == 422

    def test_batch_classify_performance(self, client: TestClient, sample_clinical_diagnoses_batch):
        """Test that batch classification completes within reasonable time."""
        import time

        start_time = time.time()
        response = client.post(
            "/v1/classify/batch", json={"texts": sample_clinical_diagnoses_batch}
        )
        end_time = time.time()

        assert response.status_code == 200
        # The API call should complete in reasonable time (< 5 seconds for this test)
        assert (end_time - start_time) < 5.0


class TestClinicalDiagnosisValidation:
    """Tests specifically for clinical diagnosis data validation."""

    def test_realistic_medical_terminology(
        self, client: TestClient, clinical_diagnosis_by_category
    ):
        """Test that realistic medical terminology is properly processed."""
        # Test with various medical terms from different specialties
        medical_texts = [
            clinical_diagnosis_by_category["neoplasms"][0],  # Contains "malignant neoplasm"
            clinical_diagnosis_by_category["cardiovascular_diseases"][
                0
            ],  # Contains "echocardiography"
            clinical_diagnosis_by_category["nervous_system_diseases"][0],  # Contains "MRI"
        ]

        response = client.post("/v1/classify/batch", json={"texts": medical_texts})

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3

    def test_special_characters_in_diagnosis(
        self, client: TestClient, special_characters_diagnosis
    ):
        """Test that special characters in medical text are handled correctly."""
        response = client.post("/v1/classify", json={"text": special_characters_diagnosis})

        assert response.status_code == 200
        data = response.json()
        assert "label" in data

    def test_multilingual_diagnosis(self, client: TestClient, multilingual_diagnosis):
        """Test that multilingual content is processed."""
        response = client.post("/v1/classify", json={"text": multilingual_diagnosis})

        assert response.status_code == 200
        data = response.json()
        assert "label" in data

    def test_diagnosis_with_numbers(self, client: TestClient):
        """Test that diagnoses with numerical data are processed."""
        diagnosis = "Patient with Stage 3 cancer, T2N1M0, requiring chemotherapy."
        response = client.post("/v1/classify", json={"text": diagnosis})

        assert response.status_code == 200
        data = response.json()
        assert "label" in data

    def test_diagnosis_with_measurements(self, client: TestClient):
        """Test that diagnoses with medical measurements are processed."""
        diagnosis = "Blood pressure 140/90 mmHg, heart rate 85 bpm, temperature 38.5°C."
        response = client.post("/v1/classify", json={"text": diagnosis})

        assert response.status_code == 200
        data = response.json()
        assert "label" in data


class TestResponseConsistency:
    """Tests for response consistency and reliability."""

    def test_same_input_same_output(self, client: TestClient, sample_clinical_diagnosis):
        """Test that the same input produces consistent output."""
        response1 = client.post("/v1/classify", json={"text": sample_clinical_diagnosis})
        response2 = client.post("/v1/classify", json={"text": sample_clinical_diagnosis})

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # The label should be consistent
        assert data1["label"] == data2["label"]

    def test_batch_vs_single_consistency(self, client: TestClient, sample_clinical_diagnosis):
        """Test that batch and single endpoints produce consistent results."""
        # Single endpoint
        single_response = client.post("/v1/classify", json={"text": sample_clinical_diagnosis})

        # Batch endpoint with same item
        batch_response = client.post(
            "/v1/classify/batch", json={"texts": [sample_clinical_diagnosis]}
        )

        assert single_response.status_code == 200
        assert batch_response.status_code == 200

        single_data = single_response.json()
        batch_data = batch_response.json()

        # The label should be consistent
        assert single_data["label"] == batch_data["results"][0]["label"]

    def test_model_version_present(self, client: TestClient, sample_clinical_diagnosis):
        """Test that model version is always present in responses."""
        single_response = client.post("/v1/classify", json={"text": sample_clinical_diagnosis})
        batch_response = client.post(
            "/v1/classify/batch", json={"texts": [sample_clinical_diagnosis]}
        )

        assert single_response.status_code == 200
        assert batch_response.status_code == 200

        single_data = single_response.json()
        batch_data = batch_response.json()

        assert "model_version" in single_data
        assert "model_version" in batch_data
        assert isinstance(single_data["model_version"], str)
        assert isinstance(batch_data["model_version"], str)

    def test_inference_time_reasonable(self, client: TestClient, sample_clinical_diagnoses):
        """Test that inference time is within reasonable bounds."""
        response = client.post("/v1/classify/batch", json={"texts": sample_clinical_diagnoses})

        assert response.status_code == 200
        data = response.json()

        # Inference time should be positive and reasonable
        assert data["inference_ms"] > 0
        assert data["inference_ms"] < 10000  # Less than 10 seconds
