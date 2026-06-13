# Metrics Documentation for Recommender Systems

## Overview

This document explains the metrics used for evaluating recommender models, their interpretation, and why multiple metrics are necessary for a complete assessment.

## Metrics Tracked

All experiments track the following metrics:

1. **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve)
2. **Average Precision** (AP)
3. **Hit Rate@K** (HR@K)
4. **NDCG@K** (Normalized Discounted Cumulative Gain at K)

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

**Best practice:** Use all metrics together to get a complete picture of model performance. Don't optimize for a single metric.

## Current Implementation

In our experiments:
- All four metrics are tracked and logged to MLflow
- Early stopping can be configured to monitor either AUC-ROC or NDCG@10
- NDCG@10 is used for early stopping by default (better for ranking tasks)
- Metrics are computed at the end of training on the validation set

## Metric Calculation

- **AUC-ROC and AP:** Computed during training on the validation set using sklearn functions
- **Hit Rate@K and NDCG@K:** Computed at the end of training using sampled evaluation for efficiency
- K=10 is used for ranking metrics (configurable via `ranking_k` parameter)

## References

- [Evaluating Recommender Systems](https://dl.acm.org/doi/10.1145/3197390)
- [AUC is not a good metric for ranking](https://towardsdatascience.com/auc-roc-and-auc-pr-for-imbalanced-classification-why-you-should-never-use-one-without-the-other-422707532531)
