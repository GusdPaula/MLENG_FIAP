# ML Recommender System — Todo List

## Completed Items ✅
- [x] Improve messy_function into pieces and modules
- [x] Understand getLogger and logging best practices

## High Priority 🔴

### Model Factory Bug (Critical)
- [x] Fix ModelFactory to filter hyperparameters per model type
  - Current: Passes all hyperparams to all models (causes errors)
  - Fix: Implement MODEL_PARAM_MAP with allowed params per model
  - Example: GMF should only accept `embedding_dim`, not `hidden_layers`

### Training Strategy Improvements
- [x] Change early stopping from AUC to NDCG@10
  - Current: Early stopping monitors AUC
  - Fix: Monitor NDCG@10 or HitRate@K for better ranking

### MLflow Tracking Improvements
- [ ] Log models explicitly with `mlflow.pytorch.log_model()`
- [ ] Register best models using `mlflow.register_model()`
- [ ] Ensure consistent metric logging per experiment
- [ ] Track loss curves in MLflow
- [ ] Deploy MLflow experiments to production environment

## Medium Priority 🟡

### Git & Model Artifact Tracking
- [x] Fix .gitignore to properly handle model artifacts
  - Use unignore hierarchy for mlflow_experiments
  - Keep specific .pt models in Git as needed
  - Use MLflow or DVC for experiment tracking
- [x] Fix .pre-commit-config.yaml to ensure hooks are triggered in commits

### DVC Integration
- [x] Better use DVC for data versioning
- [x] Better use DVC for experiment tracking
- [x] Ensure processed datasets are tracked with DVC

### Testing & Code Quality
- [x] Create unit tests to cover more than 80% of the codebase
  - Identify untested modules and functions
  - Add comprehensive test coverage
  - Ensure tests pass in CI/CD pipeline

### Loss Function Improvements
- [ ] Research and implement BPR loss (pairwise ranking)
- [ ] Consider contrastive loss (InfoNCE)
- [ ] Evaluate LambdaRank-style losses

### Negative Sampling Improvements
- [ ] Implement hard negative mining
- [ ] Add popularity-aware negatives
- [ ] Consider in-batch negatives

## Low Priority 🟢

### Documentation & Style
- [x] Ensure all comments use Google docstring style
- [x] Review and standardize docstrings across all modules
- [x] Add type hints where missing

### Model Evaluation
- [x] Ensure all experiments track: AUC-ROC, Average Precision, HitRate@K, NDCG@K
- [x] Avoid optimizing only one metric
- [x] Document metric interpretation (AUC ≠ good recommender)

### Architecture
- [x] Review separation of concerns (config, model, training, tracking)
  - Already partially addressed with refactoring
  - Verify clean boundaries between layers
- [x] Check for modularity and repetitive code
  - Identify duplicate logic across modules
  - Extract common patterns into reusable functions
  - Ensure DRY principle is followed

## Notes

### Key Insights from Improvements Document
- High AUC (~0.91) with low NDCG (~0.07) = good classifier, bad ranking
- Lower AUC (~0.87) with better NDCG (~0.30) = better recommender system
- NDCG@K is usually more important for recommendation systems
- HitRate@K reflects user experience better than AUC

### Current Status
- Refactoring completed: modular code structure in place
- Training pipeline updated to use new modules
- Logging migrated from print to logger
- Ready to implement model-specific improvements
