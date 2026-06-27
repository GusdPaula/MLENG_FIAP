# Tech Challenge Fase 02 - Sistema de Recomendação de E-commerce

## Descrição do Projeto

Este projeto implementa um sistema de recomendação de e-commerce utilizando técnicas de aprendizado profundo e aprendizado de máquina tradicional. O sistema foi desenvolvido como parte do Tech Challenge Fase 02 da FIAP.

## Arquitetura do Sistema

O projeto utiliza uma arquitetura baseada em pipelines DVC para orquestrar o fluxo de trabalho:

1. **Pré-processamento**: Transformação de eventos brutos em interações processadas
2. **Treinamento**: Treinamento de modelos de recomendação
3. **Avaliação**: Comparação de modelos treinados com baselines

## Modelos Implementados

### Modelos de Aprendizado Profundo (PyTorch)

- **NCF (Neural Collaborative Filtering)**: Combina colaboração neural com fatoração de matrizes
- **GMF (Generalized Matrix Factorization)**: Fatoração de matrizes generalizada
- **MatrixFactorization**: Fatoração de matrizes clássica

### Modelos Baseline

- **PopularityRecommender**: Recomenda itens mais populares
- **LogisticRegressionRecommender**: Regressão logística para classificação binária

## Estratégias de Processamento de Dados

- **Weighted**: Utiliza pesos baseados na frequência de interações
- **Binary**: Converte interações em valores binários (0/1)
- **Implicit**: Foco em feedback implícito (visualizações, cliques, etc.)

## Estrutura do Projeto

```
fase_2/
├── ecommerce_recommender/
│   ├── configs/
│   │   ├── model.yaml          # Configuração do modelo
│   │   └── mlflow.yaml         # Configuração do MLflow
│   ├── data/
│   │   ├── raw/                # Dados brutos
│   │   └── processed/          # Dados processados
│   ├── models/                 # Modelos treinados
│   ├── reports/                # Métricas de avaliação
│   └── src/recommender/
│       ├── data/               # Processamento de dados
│       ├── models/             # Implementação dos modelos
│       ├── pipelines/          # Pipelines DVC
│       ├── training/           # Lógica de treinamento
│       └── mlflow_toolkit/     # Integração com MLflow
├── infra/                      # Infraestrutura Terraform (AWS)
└── dvc.yaml                    # Configuração do pipeline DVC
```

## Requisitos

- Python 3.12+
- Poetry para gerenciamento de dependências
- DVC para versionamento de dados
- MLflow para tracking de experimentos
- CUDA (opcional, para treinamento em GPU)

## Instalação

```bash
# Clone o repositório
cd fase_2

# Instale as dependências
poetry install

# Configure o AWS CLI (para DVC e MLflow S3)
aws configure --profile aws
```

## Configuração

### DVC

```bash
# Configure o remote S3
dvc remote add s3-public s3://fiap-ml-dvc-bucket-tech-challenger
dvc remote modify s3-public profile aws
```

### MLflow

O arquivo `configs/mlflow.yaml` configura o tracking do MLflow:

```yaml
mlflow:
  tracking_uri: https://mlflow.asgardprint.com.br
  experiment_name: ecommerce_recommender_fiap_5
```

## Execução do Pipeline

### Pipeline Completo DVC

```bash
cd fase_2
dvc repro
```

O arquivo [`dvc.yaml`](dvc.yaml) define a estrutura do pipeline DVC com os estágios de pré-processamento, treinamento e avaliação.

### Pipeline Individual

#### Pré-processamento

```bash
cd ecommerce_recommender
poetry run python -c "import sys; sys.path.insert(0, 'src'); from recommender.pipelines.preprocess_pipeline import run_preprocess_pipeline; run_preprocess_pipeline()"
```

#### Treinamento (Modelo Único)

```bash
cd ecommerce_recommender
poetry run python -c "import sys; sys.path.insert(0, 'src'); from recommender.pipelines.train_pipeline import run_training_pipeline; run_training_pipeline()"
```

#### Treinamento (Modo Compreensivo)

Treina todas as combinações (3 modelos × 3 processadores) + baselines:

```bash
cd ecommerce_recommender
poetry run python -c "import sys; sys.path.insert(0, 'src'); from recommender.pipelines.train_pipeline import run_training_pipeline; run_training_pipeline(comprehensive=True)"
```

#### Avaliação

```bash
cd ecommerce_recommender
poetry run python -c "import sys; sys.path.insert(0, 'src'); from recommender.pipelines.evaluate_pipeline import run_evaluation_pipeline; run_evaluation_pipeline()"
```

## Configuração do Modelo

O arquivo `configs/model.yaml` permite configurar:

- Tipo de modelo (ncf, gmf, matrix_factorization)
- Hiperparâmetros (embedding_dim, hidden_layers, dropout)
- Estratégia de processamento (weighted, binary, implicit)
- Early stopping
- Batch size, learning rate, epochs

## Métricas de Avaliação

O sistema avalia os modelos utilizando:

- **AUC-ROC**: Área sob a curva ROC - mede a capacidade do modelo de distinguir entre classes positivas e negativas
- **Average Precision**: Precisão média - média das precisões em cada recall
- **Hit Rate@K**: Taxa de acerto no top-K - proporção de usuários para os quais pelo menos um item relevante aparece no top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain - mede a qualidade do ranking considerando a posição dos itens relevantes
- **Precision@K**: Precisão no top-K - proporção de itens relevantes entre os top-K recomendados
- **Recall@K**: Recall no top-K - proporção de itens relevantes que aparecem no top-K
- **MRR@K**: Mean Reciprocal Rank - média dos rankings inversos do primeiro item relevante

Para documentação detalhada sobre as métricas, consulte o [METRICS_DOCUMENTATION.md](METRICS_DOCUMENTATION.md).

## Design Patterns Utilizados

O projeto utiliza diversos padrões de design para garantir código limpo e manutenível:

### Factory Pattern
- **ModelFactory**: Cria instâncias de modelos de recomendação (NCF, GMF, MatrixFactorization) baseado em configuração
- Abstrai a lógica de criação de modelos complexos
- Permite fácil adição de novos modelos

### Strategy Pattern
- **DataProcessorContext**: Define diferentes estratégias de processamento de dados (weighted, binary, implicit)
- Permite alternar entre estratégias em tempo de execução
- Cada estratégia encapsula um algoritmo de processamento diferente

### Template Method Pattern
- **Trainer**: Define o esqueleto do algoritmo de treinamento
- Subclasses ou métodos específicos implementam passos variáveis (train_epoch, evaluate)
- Permite personalização sem alterar a estrutura do algoritmo

### Singleton Pattern
- **Settings**: Configurações globais carregadas uma única vez
- Garante consistência de configurações em toda a aplicação
- Evita carregamento múltiplo de variáveis de ambiente

### Builder Pattern
- **MLflowToolkit**: Constrói configurações complexas de tracking de experimentos
- Permite configuração passo a passo do MLflow
- Facilita testes e configurações diferentes

## Tracking de Experimentos

Todos os experimentos são rastreados no MLflow:

- Parâmetros de treinamento
- Métricas de avaliação
- Modelos treinados
- Datasets utilizados

### Servidor MLflow

Acesse o servidor MLflow para visualizar todos os experimentos:
- URL: https://mlflow.asgardprint.com.br
- Experimento: `ecommerce_recommender_fiap_5`

O servidor permite:
- Comparação entre diferentes runs
- Visualização de métricas em tempo real
- Download de modelos treinados
- Registro de modelos para produção

### Model Card

Documentação detalhada dos modelos disponível no Model Card:
- [Model Card](ecommerce_recommender/models/model_card.md)
- Descrição da arquitetura
- Hiperparâmetros utilizados
- Performance em diferentes métricas
- Limitações e considerações de uso
- Dados de treinamento utilizados

Consulte o Model Card para informações detalhadas sobre cada modelo treinado.

### Vídeo de Apresentação

Vídeo de apresentação do projeto disponível no STAR:
- Demonstração do sistema de recomendação
- Explicação da arquitetura
- Resultados obtidos
- Comparação entre modelos

[Link do vídeo - Placeholder]

### Slides

Slides de apresentação do projeto:
- Contexto do problema
- Solução proposta
- Arquitetura técnica
- Resultados e conclusões

[Link dos slides - Placeholder]

## Infraestrutura

O projeto utiliza Terraform para provisionar:

- Bucket S3 para artefatos MLflow
- Bucket S3 para dados DVC
- IAM roles para permissões necessárias

## Desenvolvimento

### Estrutura de Código

O código segue princípios de Clean Code:

- Classes com métodos ≤ 20 linhas
- Separação de responsabilidades
- Padrões de design (Factory, Strategy, Template Method)

### Pipelines Refatorados

Todos os pipelines foram refatorados para classes:

- `PreprocessPipeline`: Pré-processamento de dados
- `TrainingPipeline`: Treinamento de modelos
- `EvaluationPipeline`: Avaliação de modelos

## Contribuição

Este é um projeto acadêmico para o Tech Challenge FIAP. Para contribuições:

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Licença

Este projeto é parte do Tech Challenge FIAP Fase 02.
