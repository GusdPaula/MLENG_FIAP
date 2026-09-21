"""Testa o contrato de dados Pandera — valida regras rígidas."""

import pandas as pd
import pandera.pandas as pa
import pytest

from credit_scoring.data.contracts import validate, validate_safe


class TestContractValid:
    def test_reference_dataset_passes(self, reference_df):
        result = validate(reference_df)
        assert len(result) == len(reference_df)

    def test_validate_safe_returns_no_errors(self, reference_df):
        df_val, errors = validate_safe(reference_df)
        assert df_val is not None
        assert errors == []


class TestContractViolations:
    def _make_valid_row(self) -> dict:
        return {
            "idade": 30,
            "anos_emprego": 5,
            "renda_mensal": 3000.0,
            "valor_emprestimo": 10000.0,
            "score_credito": 650.0,
            "historico_pagamentos": "bom",
            "finalidade": "pessoal",
            "possui_conta_poupanca": 1,
            "inadimplente": 0,
        }

    def test_idade_menor_que_18_falha(self):
        row = self._make_valid_row()
        row["idade"] = 17
        df = pd.DataFrame([row])
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            validate(df)

    def test_renda_nula_falha(self):
        row = self._make_valid_row()
        df = pd.DataFrame([row])
        df["renda_mensal"] = None
        _, errors = validate_safe(df)
        assert len(errors) > 0

    def test_renda_negativa_falha(self):
        row = self._make_valid_row()
        row["renda_mensal"] = -500.0
        df = pd.DataFrame([row])
        _, errors = validate_safe(df)
        assert len(errors) > 0

    def test_score_fora_do_range_falha(self):
        row = self._make_valid_row()
        row["score_credito"] = 1200.0  # > 900
        df = pd.DataFrame([row])
        _, errors = validate_safe(df)
        assert len(errors) > 0

    def test_valor_emprestimo_zero_falha(self):
        row = self._make_valid_row()
        row["valor_emprestimo"] = 0.0
        df = pd.DataFrame([row])
        _, errors = validate_safe(df)
        assert len(errors) > 0

    def test_duplicatas_falham(self):
        row = self._make_valid_row()
        df = pd.concat([pd.DataFrame([row]), pd.DataFrame([row])], ignore_index=True)
        _, errors = validate_safe(df)
        assert len(errors) > 0

    def test_producao_com_violacoes_bloqueada(self, production_df_with_violations):
        df_val, errors = validate_safe(production_df_with_violations)
        assert df_val is None
        assert len(errors) > 0
