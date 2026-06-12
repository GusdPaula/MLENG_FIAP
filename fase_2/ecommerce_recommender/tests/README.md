# Tests

This directory contains the automated tests for the recommender project.

## Structure

```text
tests/
  unit/
  integration/
```

## Unit tests

The `unit/` folder checks small pieces of the codebase in isolation:

- model forward passes and output ranges
- model factory registration
- dataset loading and negative sampling
- data processor strategies
- data pipeline helpers
- Kaggle and BigQuery helpers
- MLflow toolkit behavior

## Integration tests

The `integration/` folder is reserved for broader end-to-end checks that combine multiple parts of the system.

## Notes

- Most tests use lightweight fakes or temporary files so they can run without external services.
- Some tests simulate optional dependencies such as BigQuery or Kaggle modules.
- The project currently uses `pytest` as the test runner.

## How to run the tests

### With `pip`

```bash
PYTHONPATH=src python -m pytest tests/unit -v
PYTHONPATH=src python -m pytest tests/integration -v
```

### With Poetry

```bash
poetry run pytest tests/unit -v
poetry run pytest tests/integration -v
```

If your environment does not automatically expose `src/`, prefix the commands with `PYTHONPATH=src`.
