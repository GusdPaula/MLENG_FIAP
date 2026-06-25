# Testes

Este diretório contém os testes automatizados para o projeto de recomendação.

## Estrutura

```text
tests/
  unit/
  integration/
```

## Testes unitários

A pasta `unit/` verifica pequenas partes do código em isolamento:

- passes forward do modelo e intervalos de saída
- registro da factory de modelos
- carregamento de dataset e negative sampling
- estratégias de processador de dados
- auxiliares de pipeline de dados
- auxiliares do Kaggle e BigQuery
- comportamento do toolkit MLflow

## Testes de integração

A pasta `integration/` é reservada para verificações mais abrangentes end-to-end que combinam múltiplas partes do sistema.

## Notas

- A maioria dos testes usa fakes leves ou arquivos temporários para que possam rodar sem serviços externos.
- Alguns testes simulam dependências opcionais como módulos do BigQuery ou Kaggle.
- O projeto atualmente usa `pytest` como executor de testes.

## Como rodar os testes

### Com `pip`

```bash
PYTHONPATH=src python -m pytest tests/unit -v
PYTHONPATH=src python -m pytest tests/integration -v
```

### Com Poetry

```bash
poetry run pytest tests/unit -v
poetry run pytest tests/integration -v
```

Se seu ambiente não expõe automaticamente `src/`, prefixe os comandos com `PYTHONPATH=src`.
