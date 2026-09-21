"""Fixtures compartilhadas para todos os testes."""

import pytest

from credit_scoring.data.generate import generate_reference_dataset
from credit_scoring.data.simulate_drift import simulate_production_drift


@pytest.fixture(scope="session")
def reference_df():
    return generate_reference_dataset(n_samples=200, seed=42)


@pytest.fixture(scope="session")
def production_df(reference_df):
    return simulate_production_drift(reference_df, seed=123, introduce_contract_violations=False)


@pytest.fixture(scope="session")
def production_df_with_violations(reference_df):
    return simulate_production_drift(reference_df, seed=123, introduce_contract_violations=True)
