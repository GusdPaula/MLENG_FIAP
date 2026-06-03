# Next Steps

1. Validate the code
   - Run unit tests and linting
   - Confirm the data pipeline scripts execute correctly
2. Set keys for GCP and Kaggle
   - Configure `GOOGLE_APPLICATION_CREDENTIALS`
   - Set Kaggle credentials for `kagglehub`
3. Add DVC to the pipeline
   - Track raw and processed datasets
   - Define DVC stages for preprocess, feature_eng, train, and evaluate
4. Perform EDA
   - Explore `events.csv`, `category_tree.csv`, and combined item properties
   - Identify feature engineering opportunities
5. Build and train the model
   - Implement recommendation model training
   - Log experiments to MLflow
