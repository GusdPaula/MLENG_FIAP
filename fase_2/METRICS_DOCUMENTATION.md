# Documentação de Métricas para Sistemas de Recomendação

## Visão Geral

Este documento explica as métricas utilizadas para avaliar modelos de recomendação, sua interpretação e por que múltiplas métricas são necessárias para uma avaliação completa.

## Métricas Rastreadas

Todos os experimentos rastreiam as seguintes métricas:

1. **AUC-ROC** (Área Abaixo da Curva Característica de Operação do Receptor)
2. **Average Precision** (AP - Precisão Média)
3. **Hit Rate@K** (HR@K - Taxa de Acertos em K)
4. **NDCG@K** (Normalized Discounted Cumulative Gain at K - Ganho Cumulativo Descontado Normalizado em K)
5. **Precision@K** (Prec@K - Precisão em K)
6. **Recall@K** (Rec@K - Revocação em K)
7. **MRR@K** (Mean Reciprocal Rank at K - Rank Recíproco Médio em K)

## Interpretação das Métricas

### AUC-ROC (Área Abaixo da Curva ROC)

**Definição:** Mede a capacidade do modelo de distinguir entre itens positivos e negativos. É a probabilidade de que um item positivo escolhido aleatoriamente seja classificado mais alto que um item negativo escolhido aleatoriamente.

**Intervalo:** 0.0 a 1.0 (quanto maior, melhor)

**Interpretação:**
- **AUC ≠ Bom Recomendador:** Um AUC-ROC alto indica boa capacidade de discriminação, mas não garante boas recomendações top-K. Um modelo pode ter AUC alto mas ainda classificar itens relevantes mal nas primeiras posições.
- AUC é uma métrica de qualidade de classificação para todo o catálogo de itens, não focada em desempenho top-K
- Bom para tarefas de classificação binária, mas menos informativo para sistemas de recomendação onde nos importamos com as principais recomendações

**Quando usar:**
- Para avaliação inicial do modelo
- Para comparar modelos na qualidade geral de classificação
- Quando a capacidade de discriminação é importante

**Limitações:**
- Não considera a posição dos itens na classificação
- Não mede quão bem os itens relevantes são posicionados no top-K
- Pode ser enganoso para sistemas de recomendação

### Average Precision (AP - Precisão Média)

**Definição:** Calcula a precisão média em cada limiar onde a revocação muda. Resume a curva precisão-revocação.

**Intervalo:** 0.0 a 1.0 (quanto maior, melhor)

**Interpretação:**
- Mede a qualidade das previsões positivas
- Considera tanto precisão quanto revocação
- Mais sensível ao desbalanceamento de classes que AUC-ROC
- Bom para cenários onde itens positivos são raros

**Quando usar:**
- Quando precisão e revocação são ambas importantes
- Para conjuntos de dados desbalanceados (poucas interações positivas)
- Quando falsos positivos são custosos

**Limitações:**
- Ainda é uma métrica global, não focada em top-K
- Não considera a qualidade da posição na classificação

### Hit Rate@K (HR@K - Taxa de Acertos em K)

**Definição:** Proporção de usuários para os quais pelo menos um item relevante aparece nas principais K recomendações.

**Intervalo:** 0.0 a 1.0 (quanto maior, melhor)

**Interpretação:**
- Mede se o modelo consegue encontrar pelo menos um item relevante no top-K
- Mede diretamente o sucesso da recomendação para usuários
- K=10 é comum para e-commerce (top 10 recomendações)
- Mais alinhado com a experiência real do usuário que AUC

**Quando usar:**
- Para avaliar qualidade de recomendação
- Quando desempenho top-K importa (maioria dos casos de uso)
- Para medir satisfação do usuário

**Limitações:**
- Não considera a posição de itens relevantes dentro do top-K
- Não mede qualidade de classificação além de "pelo menos um acerto"

### NDCG@K (Normalized Discounted Cumulative Gain at K - Ganho Cumulativo Descontado Normalizado em K)

**Definição:** Mede a qualidade da classificação considerando a posição de itens relevantes no top-K. Itens relevantes classificados mais alto contribuem mais para a pontuação.

**Intervalo:** 0.0 a 1.0 (quanto maior, melhor)

**Interpretação:**
- Considera tanto se itens relevantes estão no top-K QUANTO suas posições
- Usa desconto logarítmico: itens classificados mais alto são mais valiosos
- Normalizado pela classificação ideal, então 1.0 é perfeito
- Melhor métrica para qualidade de classificação em sistemas de recomendação

**Quando usar:**
- Para avaliar qualidade de classificação (mais importante para recomendadores)
- Quando a posição dos itens no top-K importa
- Para comparar algoritmos de classificação

**Limitações:**
- Mais computacionalmente custoso que HR@K
- Requer definir o que constitui "relevante" (geralmente binário)

### Precision@K (Prec@K - Precisão em K)

**Definição:** Proporção de itens recomendados no top-K que são realmente relevantes para o usuário.

**Intervalo:** 0.0 a 1.0 (quanto maior, melhor)

**Fórmula:** `Precision@K = |itens relevantes no top-K| / K`

**Interpretação:**
- Mede a "qualidade" da lista de recomendações
- Responde: "Dos K itens que recomendei, quantos foram bons?"
- Complementar a Recall@K — alta precisão significa poucas recomendações ruins

**Quando usar:**
- Quando mostrar itens irrelevantes tem um custo (ex: espaço de tela limitado)
- Para avaliar a densidade de itens relevantes na lista de recomendações

**Limitações:**
- Não considera a posição dentro do top-K
- Penaliza modelos igualmente por itens irrelevantes na posição 1 vs posição K

### Recall@K (Rec@K - Revocação em K)

**Definição:** Proporção de todos os itens relevantes para um usuário que aparecem nas principais K recomendações.

**Intervalo:** 0.0 a 1.0 (quanto maior, melhor)

**Fórmula:** `Recall@K = |itens relevantes no top-K| / |todos os itens relevantes para o usuário|`

**Interpretação:**
- Mede a "cobertura" de itens relevantes
- Responde: "De todos os itens que o usuário gosta, quantos eu encontrei?"
- Alta revocação significa que o modelo é bom em encontrar todos os itens relevantes

**Quando usar:**
- Quando perder um item relevante é custoso
- Para entender quão bem o modelo cobre os interesses do usuário
- Quando usuários têm múltiplos itens de interesse

**Limitações:**
- Pode ser enganoso quando usuários têm muito poucos itens relevantes
- Não considera a posição dentro do top-K

### MRR@K (Mean Reciprocal Rank at K - Rank Recíproco Médio em K)

**Definição:** Média do rank recíproco do primeiro item relevante na lista top-K através de todos os usuários.

**Intervalo:** 0.0 a 1.0 (quanto maior, melhor)

**Fórmula:** `MRR = (1/|Usuários|) * Σ (1 / rank_do_primeiro_item_relevante)`

**Interpretação:**
- Mede quão rapidamente o modelo coloca um item relevante no topo
- MRR=1.0 significa que o primeiro item é sempre relevante
- MRR=0.5 significa que o primeiro item relevante está tipicamente na posição 2
- Particularmente importante quando usuários olham apenas os primeiros itens

**Quando usar:**
- Quando o primeiro item relevante é o mais importante (ex: resultados de busca, recomendações de item único)
- Para avaliar "tempo até o primeiro item relevante"
- Quando a paciência do usuário é limitada

**Limitações:**
- Considera apenas o primeiro item relevante, ignora os subsequentes
- Pode não capturar a qualidade completa da recomendação para usuários com muitos itens relevantes

---

## Por Que Múltiplas Métricas São Necessárias

### AUC ≠ Bom Recomendador

Uma pontuação AUC-ROC alta não garante um bom sistema de recomendação porque:

1. **Global vs Local:** AUC mede a qualidade geral de classificação em todo o catálogo de itens, mas usuários veem apenas recomendações top-K.

2. **Independência de Posição:** AUC não se importa se itens relevantes estão na posição 1 ou posição 1000, desde que sejam classificados mais alto que negativos em média.

3. **Experiência do Usuário:** Usuários interagem com as principais recomendações, não com o catálogo inteiro. Um modelo com AUC alto ainda pode classificar itens relevantes mal nas primeiras posições.

### Métricas Complementares

Cada métrica fornece informações diferentes:
- **AUC-ROC:** Capacidade geral de discriminação
- **Average Precision:** Tradeoff precisão-revocação
- **Hit Rate@K:** Sucesso de recomendação focado no usuário
- **NDCG@K:** Qualidade de classificação com consciência de posição
- **Precision@K:** Qualidade/densidade de recomendações
- **Recall@K:** Cobertura de interesses do usuário
- **MRR@K:** Velocidade até o primeiro resultado relevante

**Melhor prática:** Use todas as métricas juntas para obter uma imagem completa do desempenho do modelo. Não otimize para uma única métrica.

## Implementação Atual

Em nossos experimentos:
- Todas as sete métricas são rastreadas e registradas no MLflow
- Early stopping pode ser configurado para monitorar AUC-ROC ou NDCG@10
- NDCG@10 é usado para early stopping por padrão (melhor para tarefas de classificação)
- Métricas são computadas ao final do treinamento no conjunto de validação

## Cálculo de Métricas

- **AUC-ROC e AP:** Computados durante o treinamento no conjunto de validação usando funções do sklearn
- **Hit Rate@K, NDCG@K, Precision@K, Recall@K, MRR@K:** Computados ao final do treinamento usando avaliação amostrada para eficiência
- K=10 é usado para métricas de classificação (configurável via parâmetro `ranking_k`)

## Referências

- [Evaluating Recommender Systems](https://dl.acm.org/doi/10.1145/3197390)
- [AUC is not a good metric for ranking](https://towardsdatascience.com/auc-roc-and-auc-pr-for-imbalanced-classification-why-you-should-never-use-one-without-the-other-422707532531)
