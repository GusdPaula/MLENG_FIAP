# Visão geral do pacote `recommender`

Esta pasta contém o código fonte para um pequeno projeto de sistema de recomendação construído com PyTorch.

O código é organizado em torno de três principais ideias:

1. Carregar eventos brutos de e-commerce.
2. Transformar esses eventos em amostras de treinamento.
3. Treinar um de vários modelos de recomendação através de um arquivo de configuração.

## Estrutura de pastas

```text
src/recommender/
  data/         # auxiliares de carregamento, limpeza e pré-processamento
  models/       # implementações de modelos de recomendação e factory
  training/     # loop de treinamento e métricas de avaliação
  mlflow_toolkit/ # wrapper MLflow para experimentos, datasets e modelos
  pipelines/    # orquestração end-to-end
  api/          # API de predição com detecção de desvio de modelo/dados
```

## Fluxo end-to-end

O ponto de entrada principal é [`pipelines/train_pipeline.py`](recommender/pipelines/train_pipeline.py).

Em um alto nível, o pipeline faz o seguinte:

1. Lê um arquivo de configuração YAML.
2. Carrega eventos brutos de CSV.
3. Aplica uma estratégia de processamento de dados.
4. Constrói um dataset com negative sampling.
5. Divide o dataset em conjuntos de treino e validação.
6. Cria o modelo através de uma factory.
7. Treina o modelo e computa métricas.
8. Salva o artefato do modelo treinado em disco.

## Manipulação de dados

### `load_events`

Definido em [`data/dataset.py`](recommender/data/dataset.py), esta função lê o arquivo CSV bruto em um DataFrame pandas e adiciona um peso simples de evento:

- `view` -> `1`
- `addtocart` -> `2`
- `transaction` -> `3`

### Estratégias `DataProcessor`

A lógica de pré-processamento mais flexível está em [`data/processors.py`](recommender/data/processors.py).

O projeto usa um padrão de estratégia para que o pipeline possa alternar o comportamento de pré-processamento através da configuração.

Estratégias disponíveis:

- `WeightedEventProcessor`
- `BinaryInteractionProcessor`
- `ImplicitFeedbackProcessor`

O que cada uma faz:

- `WeightedEventProcessor`: mantém todos os eventos e atribui pesos diferentes dependendo do tipo de evento.
- `BinaryInteractionProcessor`: mantém apenas eventos `addtocart` e `transaction` e os trata como interações positivas.
- `ImplicitFeedbackProcessor`: mantém todos os eventos e trata cada interação como positiva com peso `1.0`.

Cada processador também:

- constrói mapeamentos `user2idx` e `item2idx`
- cria colunas `user_idx` e `item_idx`
- opcionalmente filtra usuários/itens com poucas interações

### `RecommenderDataset`

Também em [`data/dataset.py`](recommender/data/dataset.py), a classe `RecommenderDataset` prepara amostras de treinamento usando negative sampling.

Para cada interação positiva `(user, item)`:

- adiciona uma amostra positiva com rótulo `1.0`
- adiciona `num_negatives` itens amostrados que o usuário não interagiu, cada um rotulado como `0.0`

Isso torna o problema de treinamento uma tarefa de classificação binária: o modelo aprende a prever se um par `(user, item)` é provável de ser uma interação real.

## Modelos

Todos os modelos herdam de [`models/base.py`](recommender/models/base.py).

O contrato compartilhado é simples:

- cada modelo deve implementar `forward(user_ids, item_ids)`
- cada modelo deve expor um `model_name`

### 1. Matrix Factorization

Definido em [`models/matrix_factorization.py`](recommender/models/matrix_factorization.py).

Este é o baseline clássico de collaborative filtering.

Ele aprende:

- um embedding de usuário
- um embedding de item
- um bias de usuário
- um bias de item
- um bias global

O score é computado como:

```text
score(u, i) = global_bias + user_bias + item_bias + dot(user_embedding, item_embedding)
```

Então o score é passado através de um sigmoid para que a saída esteja em `[0, 1]`.

Por que é útil:

- simples
- rápido
- forte baseline para tarefas de recomendação

### 2. GMF

Definido em [`models/gmf.py`](recommender/models/gmf.py).

GMF significa Generalized Matrix Factorization.

Ele aprende embeddings de usuário e item, combina-os com multiplicação elemento a elemento, opcionalmente projeta essa representação para outro tamanho, e então prediz com uma camada linear final.

Em termos simples:

- embedding de usuário + embedding de item
- produto elemento a elemento
- projeção opcional
- dropout
- saída linear
- sigmoid

Por que é útil:

- mais flexível que matrix factorização simples
- ainda relativamente leve
- funciona bem como uma versão neural de collaborative filtering

### 3. NCF

Definido em [`models/ncf.py`](recommender/models/ncf.py).

NCF significa Neural Collaborative Filtering.

Este modelo:

- aprende um embedding de usuário
- aprende um embedding de item
- concatena ambos os embeddings
- envia-os através de uma MLP
- aplica sigmoid para produzir o score final

Os tamanhos das camadas ocultas são configuráveis através de `hidden_layers`.

Por que é útil:

- pode aprender interações usuário-item mais complexas que MF ou GMF
- tipicamente o modelo mais expressivo neste pacote

## Factory de modelos

O arquivo [`models/factory.py`](recommender/models/factory.py) contém `ModelFactory`.

A factory é um registro que mapeia um nome de string para uma classe de modelo.

Chaves de modelos embutidos atuais:

- `ncf`
- `gmf`
- `matrix_factorization`

Isso é o que torna o pipeline de treinamento orientado por configuração:

```yaml
model:
  type: ncf
```

Se você quiser adicionar um novo modelo:

1. Crie uma subclasse de `BaseRecommenderModel`.
2. Registre-a em `ModelFactory`.
3. Selecione-a no arquivo de configuração.

## Treinamento

O loop de treinamento está em [`training/trainer.py`](recommender/training/trainer.py).

O que o trainer faz:

- usa `BCELoss`
- otimiza com Adam
- treina uma epoch por vez
- avalia com:
  - ROC AUC
  - Average Precision

Detalhe importante:

- os modelos produzem probabilidades sigmoid
- por isso, `BCELoss` é uma escolha natural aqui

## Toolkit MLflow

O helper MLflow está em [`mlflow_toolkit/toolkit.py`](recommender/mlflow_toolkit/toolkit.py).

Ele é responsável por tarefas específicas do MLflow:

- configurar o tracking URI
- selecionar ou criar um experimento
- iniciar runs
- registrar parâmetros e métricas
- registrar datasets
- registrar e registrar modelos PyTorch

Isso mantém o pipeline de treinamento limpo e evita misturar código MLflow com lógica de modelo.

## Métricas de ranking

[`training/metrics.py`](recommender/training/metrics.py) adiciona métricas de estilo de recomendação:

- `hit_rate_at_k`
- `ndcg_at_k`

Estas são computadas no conjunto de validação após o treinamento.

Elas respondem a uma pergunta diferente de AUC/AP:

- AUC/AP medem a qualidade de classificação
- Hit Rate e NDCG medem o quão boas são as recomendações top ranqueadas

## Configuração do pipeline

O pipeline espera um arquivo YAML com uma seção `model`.

Valores típicos usados pelo código:

- `seed`
- `raw_events_path`
- `processor`
- `processor_kwargs`
- `min_interactions`
- `num_negatives`
- `batch_size`
- `epochs`
- `learning_rate`
- `type`
- `hyperparams`
- `artifact_dir`

Exemplo de formato:

```yaml
model:
  seed: 42
  raw_events_path: data/raw/events.csv
  processor: weighted
  min_interactions: 1
  num_negatives: 4
  batch_size: 256
  epochs: 10
  learning_rate: 0.001
  type: ncf
  hyperparams:
    embedding_dim: 64
    hidden_layers: [128, 64, 32]
    dropout: 0.2
  artifact_dir: models
```

## Artefato de saída

Ao final do treinamento, o pipeline salva um arquivo `.pt` contendo:

- o tipo do modelo
- os pesos do modelo treinado
- `user2idx`
- `item2idx`
- a configuração de treinamento
- métricas de validação

Isso torna possível recarregar o modelo mais tarde com o mesmo mapeamento usuário/item.

## Notas de implementação

- O pipeline importa as APIs públicas do pacote de `recommender.data`, `recommender.models` e `recommender.training`.
- `create_interaction_matrix` está disponível em [`data/dataset.py`](recommender/data/dataset.py), mas o pipeline principal atualmente usa as estratégias de processador.
- O código é construído em torno de recomendação de estilo feedback implícito, não um sistema completo de classificação explícita.

## Em uma frase

Este pacote carrega logs de interação de e-commerce, converte-os em pares de treinamento usuário-item, treina um modelo de recomendação configurável e armazena o artefato treinado com os mapeamentos necessários para inferência posterior.
