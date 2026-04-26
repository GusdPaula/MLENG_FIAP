# src

Este diretório contém o código principal do projeto de previsão de churn.

Componentes principais:
- `config.py`: configurações e leitura de parâmetros do ambiente.
- `logging_config.py`: configuração de logging para o projeto.
- `api/`: definição da API que expõe o serviço de inferência.
- `data/`: carregamento e pré-processamento dos dados.
- `evaluation/`: métricas e validação de desempenho do modelo.
- `models/`: treinamento, inferência e artefatos do modelo.

O pacote `src` é a base do aplicativo e organiza a lógica de dados, modelo e serviço.
