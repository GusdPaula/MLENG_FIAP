# Data Pipeline

Este módulo `data_pipeline` contém a lógica para baixar dados de e-commerce do Kaggle e carregá-los no Google BigQuery.

## Módulos

- `kaggle_data_loader.py`
  - `KaggleDataLoader`: baixa o conjunto de dados do Kaggle usando `kagglehub.dataset_download`.
  - `combine_item_properties()`: combina `item_properties_part1.csv` e `item_properties_part2.csv` em um único arquivo `item_properties.csv`.
  - `collect_files()`: retorna os arquivos CSV esperados para upload.

- `bigquery_uploader.py`
  - `BigQueryUploader`: conecta ao Google BigQuery e envia arquivos CSV.
  - `ensure_dataset()`: cria o dataset de destino, se ele ainda não existir.
  - `upload_csv()`: envia um arquivo CSV para a tabela especificada.
  - `upload_files()`: envia múltiplos arquivos e retorna os identificadores completos das tabelas no BigQuery.

- `pipeline.py`
  - `DataPipeline`: orquestra o fluxo completo baixando dados do Kaggle, combinando propriedades de itens e carregando os arquivos no BigQuery.

- `run_pipeline.py`
  - Entrada executável para o pipeline.
  - Suporta argumentos de linha de comando e variáveis de ambiente para configuração.

- `bigquery_query.py`
  - `BigQueryQuery`: executa consultas SQL no BigQuery, exporta os resultados para arquivos CSV locais e versiona esses arquivos com DVC.

## Uso

A partir do diretório `ecommerce_recommender/data_pipeline`, execute:

```bash
python run_pipeline.py \
  --gcp-project SEU_PROJETO_GCP \
  --gcp-dataset SEU_DATASET_GCP \
  --location US \
  --table-prefix ecommerce
```

Também é possível consultar o BigQuery diretamente e versionar os dados extraídos com DVC usando `BigQueryQuery`.

Exemplo de uso em Python:

```python
from pathlib import Path
from bigquery_query import BigQueryQuery

query_client = BigQueryQuery(
    project_id="SEU_PROJETO_GCP",
    dataset_id="SEU_DATASET_GCP",
    output_dir=Path("./data/exports"),
    dvc_repo_path=Path("../.."),
)

csv_path = query_client.extract_table(
    table_name="events",
    destination_name="events_export.csv",
)
print(f"Exportado e versionado: {csv_path}")
```

O CSV exportado é adicionado ao DVC com `dvc add`.

> Observação: o DVC deve estar inicializado na raiz do repositório antes de executar as exportações.
> Se o repositório não estiver no Git, use `dvc init --no-scm`.

Também é possível configurar o pipeline usando variáveis de ambiente:

- `KAGGLE_DATASET`
- `GCP_PROJECT`
- `BIGQUERY_DATASET` ou `GCP_DATASET`
- `BIGQUERY_LOCATION` ou `GCP_REGION`
- `TABLE_PREFIX`

## Arquivos de dados

O pipeline espera que os seguintes arquivos existam após o download do dataset do Kaggle:

- `category_tree.csv`
- `events.csv`
- `item_properties_part1.csv`
- `item_properties_part2.csv`

As duas partes de `item_properties` são combinadas em `item_properties.csv` antes do upload.

## Observações

- O cliente do BigQuery usa autenticação padrão do Google Cloud.
- Este módulo é propositalmente pequeno para facilitar testes independentes.
