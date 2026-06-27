# Model Card — Sistema de Recomendação E-Commerce

## Visão Geral

| Campo | Descrição |
|-------|-----------|
| **Nome** | E-Commerce Recommender System |
| **Tipo** | Sistema de recomendação baseado em filtragem colaborativa com redes neurais |
| **Framework** | PyTorch |
| **Dataset** | RetailRocket E-Commerce Dataset (Kaggle) |
| **Licença** | Projeto acadêmico (Pós Tech FIAP — MLEng Fase 2) |

O sistema recomenda produtos para usuários de um e-commerce com base em seu histórico de interações (visualizações, adições ao carrinho e compras).

---

## Uso Pretendido

**Para que serve:**
- Gerar listas de top-K produtos recomendados para cada usuário
- Comparar abordagens neurais (deep learning) com baselines clássicos

**Quem usa:**
- Equipes de data science avaliando modelos de recomendação
- Contexto acadêmico de avaliação do Tech Challenge

**Cenários fora do escopo:**
- Produção real sem retreino com dados atualizados
- Recomendação em tempo real com latência < 100ms (não otimizado para serving)
- Usuários ou itens novos sem histórico (cold start)

---

## Dataset

| Campo | Valor |
|-------|-------|
| **Fonte** | [RetailRocket E-Commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) |
| **Tipo de dados** | Interações user-item (view, addtocart, transaction) |
| **Período** | ~4.5 meses de dados de navegação |
| **Tamanho** | ~2.7M eventos, ~1.4M usuários, ~235K itens |
| **Filtro aplicado** | Mínimo de 5 interações por usuário |

**Pré-processamento:**
- Estratégia de pesos: `weighted` (view=1, addtocart=2, transaction=3)
- Negative sampling: 4 amostras negativas por interação positiva
- Split: 80% treino / 20% validação (seed fixa = 42)
- Mapeamento user/item para índices contíguos (user2idx, item2idx)

---

## Arquitetura dos Modelos

### NCF — Neural Collaborative Filtering (modelo principal)

Baseado em He et al. (2017). Concatena embeddings de usuário e item e passa por uma MLP com camadas ocultas configuráveis.

```
User Embedding (64d) ──┐
                       ├── Concat ── Linear(128) ── ReLU ── Dropout(0.2)
Item Embedding (64d) ──┘                ── Linear(64) ── ReLU ── Dropout(0.2)
                                        ── Linear(32) ── ReLU ── Dropout(0.2)
                                        ── Linear(1) ── Sigmoid ── Score [0,1]
```

### GMF — Generalized Matrix Factorization

Branch element-wise do paper NCF. Calcula o produto de Hadamard entre embeddings de usuário e item, seguido de projeção linear e Sigmoid.

```
User Embedding (64d) ──┐
                       ├── Hadamard Product ── Dropout ── Linear(1) ── Sigmoid
Item Embedding (64d) ──┘
```

### MF — Matrix Factorization (Funk-SVD)

Modelo clássico de fatoração de matrizes: `score(u, i) = μ + b_u + b_i + p_u · q_i`

```
User Embedding + User Bias ──┐
                             ├── Dot Product + Biases + Global Bias ── Sigmoid
Item Embedding + Item Bias ──┘
```

### Baselines (Scikit-Learn)

| Baseline | Descrição |
|----------|-----------|
| **Popularidade** | Pontua itens pela frequência/peso no treino (normalizado para [0,1]) |
| **Regressão Logística** | Features one-hot esparsas de user+item, sklearn LogisticRegression |

---

## Treinamento

| Parâmetro | Valor |
|-----------|-------|
| Loss | BCELoss (Binary Cross-Entropy) |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 1024 |
| Epochs | 10 |
| Early stopping | Patience=3, monitor=NDCG@10, mode=max |
| Seed | 42 (reprodutibilidade) |
| Device | CPU ou CUDA (detectado automaticamente) |

**Inicialização de pesos:**
- Embeddings: Xavier Uniform
- Camadas lineares: Kaiming Uniform (ReLU)

---

## Métricas de Avaliação

| Métrica | O que mede | Tipo |
|---------|------------|------|
| **AUC-ROC** | Capacidade de separar positivos de negativos | Classificação |
| **Average Precision** | Qualidade do ranking de probabilidades | Classificação |
| **Hit Rate@10** | "Pelo menos 1 item relevante no top-10?" | Ranking |
| **NDCG@10** | Qualidade posicional do ranking (topo > fundo) | Ranking |
| **Precision@10** | Proporção de itens relevantes no top-10 | Ranking |
| **Recall@10** | Proporção de itens relevantes encontrados no top-10 | Ranking |
| **MRR@10** | Posição do primeiro item relevante (reciprocal rank) | Ranking |

---

## Resultados

| Modelo | Loss | AUC-ROC | Avg Prec | HR@10 | NDCG@10 | Prec@10 | Rec@10 | MRR@10 | TrainLat(s) |
|--------|------|---------|----------|-------|---------|---------|--------|--------|-------------|
| **NCF Models** |
| ncf_weighted | 0.0339 | 0.9221 | 0.8446 | 0.1490 | 0.0896 | 0.0156 | 0.1518 | 0.0714 | 0.00 |
| ncf_binary | 0.0293 | 0.8576 | 0.7979 | 0.3650 | 0.3151 | 0.0590 | 0.4492 | 0.2962 | 0.00 |
| ncf_implicit | 0.0356 | 0.9202 | 0.8416 | 0.1360 | 0.0826 | 0.0143 | 0.1405 | 0.0652 | 0.00 |
| **GMF Models** |
| gmf_weighted | 0.0038 | 0.9181 | 0.8504 | 0.1840 | 0.0935 | 0.0193 | 0.1895 | 0.0647 | 0.00 |
| gmf_binary | 0.4784 | 0.7242 | 0.5066 | 0.2250 | 0.1940 | 0.0350 | 0.2551 | 0.1962 | 0.00 |
| gmf_implicit | 0.0037 | 0.9187 | 0.8514 | 0.2040 | 0.1076 | 0.0216 | 0.2147 | 0.0756 | 0.00 |
| **MF Models** |
| matrix_factorization_weighted | 0.0036 | 0.9296 | 0.8673 | 0.3390 | 0.2323 | 0.0358 | 0.3547 | 0.1958 | 0.00 |
| matrix_factorization_binary | 0.2466 | 0.8175 | 0.6718 | 0.1520 | 0.0924 | 0.0235 | 0.1509 | 0.0875 | 0.00 |
| matrix_factorization_implicit | 0.0034 | 0.9299 | 0.8671 | 0.3420 | 0.2287 | 0.0362 | 0.3563 | 0.1911 | 0.00 |
| **Baselines** |
| Popularidade | N/A | 0.8003 | 0.5528 | 0.0130 | 0.0059 | 0.0014 | 0.0121 | 0.0043 | 0.0177 |
| Regressão Logística | N/A | 0.7960 | 0.5118 | 0.0120 | 0.0060 | 0.0013 | 0.0118 | 0.0044 | 6.8681 |

**Melhor modelo:** `ncf_binary` (NDCG@10: 0.3151, HR@10: 0.3650, MRR@10: 0.2962)

*Nota: Para sistemas de recomendação, métricas de ranking (NDCG@10, HR@10) são mais relevantes que métricas de classificação (AUC-ROC).*

Os resultados são gerados automaticamente via `dvc repro` (stage evaluate) e salvos em `ecommerce_recommender/reports/metrics.json`.

---

## Limitações

| Limitação | Impacto |
|-----------|---------|
| **Cold start** | Usuários ou itens sem histórico não recebem recomendações personalizadas |
| **Dados offline** | O modelo não captura mudanças de comportamento em tempo real |
| **Catálogo estático** | Novos produtos não são incorporados sem retreino |
| **Interações implícitas** | Não captura preferência negativa explícita (só infere do que o usuário não fez) |
| **Escala de avaliação** | Métricas calculadas com sample_limit=10K e positive_limit=1K por performance |
| **Sem features de conteúdo** | Só usa IDs — não aproveita categoria, preço ou descrição dos itens |

---

## Vieses e Considerações Éticas

| Viés | Descrição | Mitigação possível |
|------|-----------|-------------------|
| **Viés de popularidade** | Itens populares são recomendados com mais frequência, reforçando sua dominância | Negative sampling, diversificação no serving |
| **Viés de exposição** | Usuários só interagem com itens que viram — ausência de interação ≠ desinteresse | Exploração (explore/exploit) no serving |
| **Viés temporal** | Dados de ~4.5 meses podem não representar sazonalidade completa | Retreino periódico |
| **Falta de diversidade** | O modelo pode criar "bolhas de filtro" recomendando itens muito similares | Métricas de diversidade (coverage, serendipity) |
| **Dados históricos** | Padrões passados podem perpetuar comportamentos que não refletem preferências atuais | Decay temporal nos pesos |

---

## Reprodutibilidade

Para reproduzir os resultados:

```bash
# 1. Instalar dependências
cd fase_2
poetry install

# 2. Obter dados versionados
dvc pull

# 3. Executar pipeline completo (preprocess → train → evaluate)
poetry run dvc repro

# 4. Resultados em:
#    - Terminal: tabela comparativa
#    - Arquivo: ecommerce_recommender/reports/metrics.json
#    - MLflow: http://localhost:5000 (se docker-compose up)
```

**Ou via Docker:**

```bash
docker compose up
```

**Garantias de reprodutibilidade:**
- Seed fixa (42) para split e inicialização de pesos
- Lock file commitado (`poetry.lock`)
- Pipeline declarativo (`dvc.yaml`)
- Dados versionados (`.dvc` files)

---

## Referências

- He, X. et al. (2017). "Neural Collaborative Filtering." WWW.
- Mitchell, M. et al. (2019). "Model Cards for Model Reporting." FAT*.
- Dataset: [RetailRocket E-Commerce](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
