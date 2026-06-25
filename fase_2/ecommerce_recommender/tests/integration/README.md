# Integration Tests for API

This directory contains integration tests for the e-commerce recommendation API.

## Test Structure

The test file `test_api_app.py` contains two types of tests:

1. **Mocked Tests**: Tests that use FastAPI's TestClient with mocked prediction service. These tests run quickly and don't require a running API server.

2. **Real API Integration Tests**: Tests that make actual HTTP requests to a running API service using the `requests` library. These tests are marked with `@pytest.mark.integration` and require the API to be running.

## Running the Tests

### Run All Tests (Mocked Only)

```bash
# From the ecommerce_recommender directory
pytest tests/integration/test_api_app.py -v
```

### Run Integration Tests Only

Integration tests require the API service to be running locally.

```bash
# Terminal 1: Start the API service
export API_KEY="default-api-key-change-in-production"
export MODEL_PATH="ecommerce_recommender/models/mlflow_experiments/gmf_binary.pt"
PYTHONPATH=src python -m api.main

# Terminal 2: Run integration tests
pytest tests/integration/test_api_app.py -v -m integration
```

### Run Specific Test Classes

```bash
# Run only health endpoint tests
pytest tests/integration/test_api_app.py::TestHealthEndpoint -v

# Run only real API health tests
pytest tests/integration/test_api_app.py::TestHealthEndpointRealAPI -v -m integration

# Run complete flow tests
pytest tests/integration/test_api_app.py::TestCompleteFlowRealAPI -v -m integration
```

## Configuration

Integration tests use environment variables for configuration:

- `API_BASE_URL`: Base URL of the API (default: `http://localhost:8000`)
- `API_KEY`: API key for authentication (default: `default-api-key-change-in-production`)

Example:
```bash
export API_BASE_URL="http://localhost:8000"
export API_KEY="default-api-key-change-in-production"
pytest tests/integration/test_api_app.py -v -m integration
```

## Test Data

The tests use test data specific to the `gmf_binary` model:
- Test user ID: `138131`
- Test item IDs: `[430292, 277119, 183411, 457231, 259078]`
- Test user ID 2: `911093`
- Test item IDs 2: `[457231, 259078, 183087]`

If using a different model (e.g., `ncf_weighted`), update these values in the `api_config` fixture.

## Test Coverage

### Endpoint Tests
- **Health Check**: `/health`
- **Model Info**: `/model/info`
- **Single Prediction**: `/predict`
- **Batch Prediction**: `/predict/batch`
- **Top-K Recommendations**: `/recommend/{user_id}`
- **Monitoring Baselines**: `/monitoring/baselines`
- **Check Shifts**: `/monitoring/check`
- **Monitoring Summary**: `/monitoring/summary`

### Scenario Tests
- **Complete Prediction Flow**: Tests the full workflow from health check to monitoring
- **Error Scenarios**: Tests authentication failures, invalid inputs, and missing data

## Starting the API for Integration Tests

### Using Python Directly

```bash
cd ecommerce_recommender
export API_KEY="default-api-key-change-in-production"
export MODEL_PATH="ecommerce_recommender/models/mlflow_experiments/gmf_binary.pt"
PYTHONPATH=src python -m api.main
```

### Using Uvicorn

```bash
cd ecommerce_recommender
export API_KEY="default-api-key-change-in-production"
export MODEL_PATH="ecommerce_recommender/models/mlflow_experiments/gmf_binary.pt"
PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Using Poetry

```bash
cd ecommerce_recommender
export API_KEY="default-api-key-change-in-production"
export MODEL_PATH="ecommerce_recommender/models/mlflow_experiments/gmf_binary.pt"
PYTHONPATH=src poetry run python -m api.main
```

## Troubleshooting

### Tests Skipped with "API service not running"

This means the integration tests cannot connect to the API service. Ensure:
1. The API service is running on the expected URL (default: `http://localhost:8000`)
2. The `API_BASE_URL` environment variable is set correctly
3. The API key matches what the service expects

### Import Errors

If you see import errors, ensure:
1. The `PYTHONPATH` includes the `src` directory
2. You're running tests from the `ecommerce_recommender` directory
3. The MVC refactoring has been completed (api module exists in src)

### Model Not Found Errors

If tests fail with model loading errors:
1. Ensure the model file exists at the path specified by `MODEL_PATH`
2. Verify the model is compatible with the current code
3. Check that the model contains the required metadata (user2idx, item2idx, etc.)

## CI/CD Integration

For CI/CD pipelines, you may want to:
1. Run mocked tests in every build (fast, no dependencies)
2. Run integration tests only when deploying or in specific environments
3. Use a Docker container to run the API service for integration tests

Example pytest configuration for CI:

```ini
# pytest.ini
[pytest]
markers =
    integration: marks tests as integration tests (deselect with '-m "not integration"')
```
