# data

Esta pasta organiza os dados usados pelo projeto.

Subpastas:
- `raw/`: dados originais de origem, sem transformações.
- `processed/`: dados já limpos e preparados para treino e inferência.

Exemplo:
- `processed/telco_churn_processed.csv`: conjunto de dados processado pronto para modelagem.

Use `src/data/loader.py` e `src/data/preprocessing.py` para carregar e transformar os dados.
