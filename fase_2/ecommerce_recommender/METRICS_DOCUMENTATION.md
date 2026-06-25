# Metrics Documentation for Recommender Systems

## Overview

This document explains the metrics used for evaluating recommender models, their interpretation, and why multiple metrics are necessary for a complete assessment.

## Metrics Tracked

All experiments track the following metrics:

1. **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve)
2. **Average Precision** (AP)
3. **Hit Rate@K** (HR@K)
4. **NDCG@K** (Normalized Discounted Cumulative Gain at K)
5. **Precision@K** (Prec@K)
6. **Recall@K** (Rec@K)
7. **MRR@K** (Mean Reciprocal Rank at K)

## Metric Interpretation

### AUC-ROC (Area Under the ROC Curve)

**Definition:** Measures the ability of the model to distinguish between positive and negative items. It's the probability that a randomly chosen positive item is ranked higher than a randomly chosen negative item.

**Range:** 0.0 to 1.0 (higher is better)

**Interpretation:**
- **AUC ≠ Good Recommender:** A high AUC-ROC indicates good discrimination ability, but it doesn't guarantee good top-K recommendations. A model can have high AUC but still rank relevant items poorly in the top positions.
- AUC is a ranking quality metric for the entire item catalog, not focused on top-K performance
- Good for binary classification tasks, but less informative for recommender systems where we care about top recommendations

**When to use:**
- For initial model assessment
- For comparing models on overall ranking quality
- When discrimination ability is important

**Limitations:**
- Doesn't account for position of items in ranking
- Doesn't measure how well relevant items are placed in top-K
- Can be misleading for recommender systems

### Average Precision (AP)

**Definition:** Computes the average precision at each threshold where the recall changes. It summarizes the precision-recall curve.

**Range:** 0.0 to 1.0 (higher is better)

**Interpretation:**
- Measures the quality of the positive predictions
- Accounts for both precision and recall
- More sensitive to class imbalance than AUC-ROC
- Good for scenarios where positive items are rare

**When to use:**
- When precision and recall are both important
- For imbalanced datasets (few positive interactions)
- When false positives are costly

**Limitations:**
- Still a global metric, not focused on top-K
- Doesn't account for ranking position quality

### Hit Rate@K (HR@K)

**Definition:** Proportion of users for whom at least one relevant item appears in the top-K recommendations.

**Range:** 0.0 to 1.0 (higher is better)

**Interpretation:**
- Measures whether the model can find at least one relevant item in the top-K
- Directly measures recommendation success for users
- K=10 is common for e-commerce (top 10 recommendations)
- More aligned with actual user experience than AUC

**When to use:**
- For evaluating recommendation quality
- When top-K performance matters (most use cases)
- For measuring user satisfaction

**Limitations:**
- Doesn't account for position of relevant items within top-K
- Doesn't measure ranking quality beyond "at least one hit"

### NDCG@K (Normalized Discounted Cumulative Gain at K)

**Definition:** Measures ranking quality by considering the position of relevant items in the top-K. Higher-ranked relevant items contribute more to the score.

**Range:** 0.0 to 1.0 (higher is better)

**Interpretation:**
- Accounts for both whether relevant items are in top-K AND their positions
- Uses logarithmic discounting: higher-ranked items are more valuable
- Normalized by the ideal ranking, so 1.0 is perfect
- Best metric for ranking quality in recommender systems

**When to use:**
- For evaluating ranking quality (most important for recommenders)
- When position of items in top-K matters
- For comparing ranking algorithms

**Limitations:**
- More computationally expensive than HR@K
- Requires defining what constitutes "relevant" (usually binary)

### Precision@K (Prec@K)

**Definition:** Proportion of recommended items in the top-K that are actually relevant to the user.

**Range:** 0.0 to 1.0 (higher is better)

**Formula:** `Precision@K = |relevant items in top-K| / K`

**Interpretation:**
- Measures the "quality" of the recommendation list
- Answers: "Of the K items I recommended, how many were good?"
- Complementary to Recall@K — high precision means few bad recommendations

**When to use:**
- When showing irrelevant items has a cost (e.g., limited screen space)
- For evaluating the density of relevant items in the recommendation list

**Limitations:**
- Doesn't account for position within top-K
- Penalizes models equally for irrelevant items at position 1 vs position K

### Recall@K (Rec@K)

**Definition:** Proportion of all relevant items for a user that appear in the top-K recommendations.

**Range:** 0.0 to 1.0 (higher is better)

**Formula:** `Recall@K = |relevant items in top-K| / |all relevant items for user|`

**Interpretation:**
- Measures the "coverage" of relevant items
- Answers: "Of all items the user likes, how many did I find?"
- High recall means the model is good at finding all relevant items

**When to use:**
- When missing a relevant item is costly
- For understanding how well the model covers user interests
- When users have multiple items of interest

**Limitations:**
- Can be misleading when users have very few relevant items
- Doesn't account for position within top-K

### MRR@K (Mean Reciprocal Rank at K)

**Definition:** Average of the reciprocal rank of the first relevant item in the top-K list across all users.

**Range:** 0.0 to 1.0 (higher is better)

**Formula:** `MRR = (1/|Users|) * Σ (1 / rank_of_first_relevant_item)`

**Interpretation:**
- Measures how quickly the model places a relevant item at the top
- MRR=1.0 means the first item is always relevant
- MRR=0.5 means the first relevant item is typically at position 2
- Particularly important when users look at the first few items only

**When to use:**
- When the first relevant item matters most (e.g., search results, single-item recommendations)
- For evaluating "time to first relevant item"
- When user patience is limited

**Limitations:**
- Only considers the first relevant item, ignores subsequent ones
- May not capture full recommendation quality for users with many relevant items

---

## Why Multiple Metrics Are Necessary

### AUC ≠ Good Recommender

A high AUC-ROC score does not guarantee a good recommender system because:

1. **Global vs Local:** AUC measures overall ranking quality across the entire item catalog, but users only see top-K recommendations.

2. **Position Independence:** AUC doesn't care if relevant items are at position 1 or position 1000, as long as they're ranked higher than negatives on average.

3. **User Experience:** Users interact with top recommendations, not the entire catalog. A model with high AUC might still rank relevant items poorly in the top positions.

### Complementary Metrics

Each metric provides different information:
- **AUC-ROC:** Overall discrimination ability
- **Average Precision:** Precision-recall tradeoff
- **Hit Rate@K:** User-focused recommendation success
- **NDCG@K:** Ranking quality with position awareness
- **Precision@K:** Quality/density of recommendations
- **Recall@K:** Coverage of user interests
- **MRR@K:** Speed to first relevant result

**Best practice:** Use all metrics together to get a complete picture of model performance. Don't optimize for a single metric.

## Current Implementation

In our experiments:
- All seven metrics are tracked and logged to MLflow
- Early stopping can be configured to monitor either AUC-ROC or NDCG@10
- NDCG@10 is used for early stopping by default (better for ranking tasks)
- Metrics are computed at the end of training on the validation set

## Metric Calculation

- **AUC-ROC and AP:** Computed during training on the validation set using sklearn functions
- **Hit Rate@K, NDCG@K, Precision@K, Recall@K, MRR@K:** Computed at the end of training using sampled evaluation for efficiency
- K=10 is used for ranking metrics (configurable via `ranking_k` parameter)

## References

- [Evaluating Recommender Systems](https://dl.acm.org/doi/10.1145/3197390)
- [AUC is not a good metric for ranking](https://towardsdatascience.com/auc-roc-and-auc-pr-for-imbalanced-classification-why-you-should-never-use-one-without-the-other-422707532531)
