import sys
sys.path.append('..')  # Add parent directory to path
import pandas as pd
import numpy as np
from db_setup.query import query_bigquery_table
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
SEED = 42


mlflow.set_tracking_uri("file:///c:/Users/vc/Documents/MLENG_FIAP/fase_1/Gustavo/mlruns")


mlflow.set_tracking_uri("file:///c:/Users/vc/Documents/MLENG_FIAP/mlruns")
mlflow.set_experiment("Telco-Churn-Prediction")


df = query_bigquery_table()
print(df.head())


df.shape


df.info()


for column in df.columns:
    empty_count = df[df[column] == ""].shape[0]
    empty_space_count = df[df[column] == " "].shape[0]
    null_count = df[df[column].isnull()].shape[0]
    nan_count = df[df[column].isna()].shape[0]
    zero_count = df[df[column] == 0].shape[0]
    if empty_count > 0:
        print(f'Empty values in {column}: {empty_count}')
    if empty_space_count > 0:
        print(f'Empty spaces in {column}: {empty_space_count}')
    if null_count > 0:
        print(f'Null values in {column}: {null_count}')
    if nan_count > 0:
        print(f'Nans in {column}: {nan_count}')
    if zero_count > 0:
        print(f'Zeros in {column}: {zero_count}')


df['SeniorCitizen'].value_counts()


df['Partner'].unique()


df['Dependents'].unique()


df['tenure'].unique()


df['PhoneService'].unique()


df['PaperlessBilling'].unique()


mean_total_charges = df[df['TotalCharges'] != ' ']['TotalCharges'].astype(float).mean()
print(f'Mean Total Charges: {mean_total_charges}')


median_total_charges = df[df['TotalCharges'] != ' ']['TotalCharges'].astype(float).median()
print(f'Median Total Charges: {median_total_charges}')


plt.hist(df[df['TotalCharges'] != ' ']['TotalCharges'].astype(float), bins=30)


df = df[df['TotalCharges'] != ' ']


df['TotalCharges'] = df['TotalCharges'].astype(float)


df['Churn'].value_counts()


df.columns


df['TotalCharges'].astype(float).describe()


plt.hist(df['MonthlyCharges'], bins=30)
plt.title('Distribution of Monthly Charges')


df['PaymentMethod'].value_counts()


df['PaperlessBilling'].value_counts()


df['Contract'].value_counts()


df['StreamingMovies'].value_counts()


df['StreamingTV'].value_counts()


df['TechSupport'].value_counts()


df['DeviceProtection'].value_counts()


df['OnlineBackup'].value_counts()


df['OnlineSecurity'].value_counts()


df['InternetService'].value_counts()


df['MultipleLines'].value_counts()


df['PhoneService'].value_counts()


df['Dependents'].value_counts()


df['Partner'].value_counts()


df['SeniorCitizen'].value_counts()


df['gender'].value_counts()


plt.hist(df['tenure'], bins=30)
plt.title('Distribution of Tenure')


df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]


df.columns = df.columns.str.replace(r'[^a-z0-9_]', '', regex=True)


df.set_index('customerid', inplace=True)


object_columns = df.select_dtypes(include=['object']).columns.tolist()
print("Object type columns:")
print(object_columns)
df_encoded = pd.get_dummies(df, columns=object_columns, drop_first=True)
print(f"\nOriginal shape: {df.shape}")
print(f"After one-hot encoding: {df_encoded.shape}")
print(f"\nNew columns: {df_encoded.columns.tolist()}")


df.info()


numerical_features = ['monthlycharges', 'totalcharges', 'tenure']


for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    plt.boxplot(df_encoded[feature])
    plt.title(f'Boxplot of {feature}')
    plt.show()


df_encoded.to_parquet("./processed_data/df_encoded.parquet")
dataset = mlflow.data.from_pandas(df_encoded, source='./processed_data/df_encoded.parquet', name="df_encoded")


X = df_encoded.drop('churn', axis=1)
X.columns = [col.replace(' ', '_') for col in X.columns]
X.columns = X.columns.str.replace(r'[^a-z0-9_]', '', regex=True)
y = df_encoded['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])


import numpy as np
def train_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    with mlflow.start_run(run_name=f'Experiment - {model_name}'):
        train_ds = mlflow.data.from_pandas(pd.concat([X_train, y_train], axis=1), 
                                       targets=y_train.name, 
                                       name="train_data")
        test_ds = mlflow.data.from_pandas(pd.concat([X_test, y_test], axis=1),
                                       targets=y_test.name,
                                       name="test_data")
        params = model.get_params()
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_input(train_ds, "train_data")
        mlflow.log_input(test_ds, "test_data")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        mlflow.log_metric("roc_auc", roc_auc)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        mlflow.log_metric("pr_auc", pr_auc)
        if hasattr(model, 'coef_'):
            feature_names = X_train.columns.tolist()
            coefficients = model.coef_[0]
            for feature_name, coef in zip(feature_names, coefficients):
                mlflow.log_param(f"feature_{feature_name}_coef", round(coef, 6))
                odds_ratio = np.exp(coef)
                mlflow.log_param(f"feature_{feature_name}_odds_ratio", round(odds_ratio, 6))
            mlflow.log_param("intercept", round(model.intercept_[0], 6))
            feature_summary = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients,
                'Odds_Ratio': np.exp(coefficients)
            }).sort_values('Coefficient', ascending=False)
            feature_summary.to_csv('./tmp/feature_coefficients.csv', index=False)
            mlflow.log_artifact('./tmp/feature_coefficients.csv', artifact_path='feature_analysis')
            print(f"\nFeature Coefficients and Odds Ratios:")
            print(feature_summary)
        mlflow.sklearn.log_model(model, name=model_name)
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{model_name}", model_name)
        print(f"\n{model.__class__.__name__} Performance:")
        print(f'Accuracy: {accuracy:.4f}')
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(classification_report(y_test, y_pred))


dummy_clf = DummyClassifier(strategy='most_frequent', random_state=SEED)
train_model(dummy_clf, X_train, y_train, X_test, y_test, model_name="Dummy Classifier")


logreg_clf = LogisticRegression(max_iter=1000, random_state=SEED)
train_model(logreg_clf, X_train, y_train, X_test, y_test, model_name="Logistic Regression")


logreg_clf = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced')
train_model(logreg_clf, X_train, y_train, X_test, y_test, model_name="Logistic Regression with Class Weight")


smote = SMOTE(random_state=SEED)
X_train_float = X_train.astype('float64')
y_train_int = y_train.astype('int')
X_train_smote, y_train_smote = smote.fit_resample(X_train_float, y_train_int)
X_train_smote = pd.DataFrame(X_train_smote, columns=X_train.columns)
logreg_clf_smote = LogisticRegression(max_iter=1000, random_state=SEED)
train_model(logreg_clf_smote, X_train_smote, y_train_smote, X_test, y_test.astype(int), model_name="Logistic Regression with SMOTE")


xgb_clf = XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss')
train_model(xgb_clf, X_train, y_train.astype(int), X_test, y_test.astype(int), model_name="XGBoost Classifier")


rf_clf = RandomForestClassifier(random_state=SEED)
train_model(rf_clf, X_train, y_train.astype(int), X_test, y_test.astype(int), model_name="Random Forest Classifier")


random_search_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb_clf_random = XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss')
random_search = RandomizedSearchCV(estimator=xgb_clf_random, param_distributions=random_search_params, n_iter=10, cv=3, verbose=2, random_state=SEED, n_jobs=-1)
random_search.fit(X_train, y_train.astype(int))
print(f"Best parameters found: {random_search.best_params_}")


best_xgb_clf = random_search.best_estimator_
train_model(best_xgb_clf, X_train, y_train.astype(int), X_test, y_test.astype(int), model_name="XGBoost Classifier with Random Search CV")


rf_random_search_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_clf_random = RandomForestClassifier(random_state=SEED)
rf_random_search = RandomizedSearchCV(estimator=rf_clf_random, param_distributions=rf_random_search_params, n_iter=10, cv=3, verbose=2, random_state=SEED, n_jobs=-1)
rf_random_search.fit(X_train, y_train.astype(int))
print(f"Best parameters found for Random Forest: {rf_random_search.best_params_}")


best_rf_clf = rf_random_search.best_estimator_
train_model(best_rf_clf, X_train, y_train.astype(int), X_test, y_test.astype(int), model_name="Random Forest Classifier with Random Search CV")


class SimpleChurnNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleChurnNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x
def train_and_evaluate_nn(X_train, y_train, X_test, y_test, model_name="Neural Network", 
                          num_epochs=100, patience=10, min_delta=0.0001, val_split=0.2,
                          learning_rate=0.001, batch_size=32, log_to_mlflow=True, NN=SimpleChurnNN):
    """
    Train and evaluate a neural network with early stopping and batch processing.
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    model_name : str
        Name of the model for logging
    num_epochs : int
        Maximum number of epochs
    patience : int
        Number of epochs with no improvement before early stopping
    min_delta : float
        Minimum improvement threshold
    val_split : float
        Validation split ratio (0.2 = 20%)
    learning_rate : float
        Learning rate for Adam optimizer
    batch_size : int
        Batch size for training and validation (default: 32)
    log_to_mlflow : bool
        Whether to log results to MLflow
    NN : class
        Neural network architecture class
    Returns:
    --------
    dict : Dictionary containing model, metrics, and training info
    """
    from torch.utils.data import DataLoader, TensorDataset
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    val_size = int(len(X_train) * val_split)
    X_train_split = X_train.iloc[:-val_size]
    X_val_split = X_train.iloc[-val_size:]
    y_train_split = y_train.iloc[:-val_size]
    y_val_split = y_train.iloc[-val_size:]
    X_train_tensor = torch.tensor(X_train_split.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_split.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_split.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_split.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if log_to_mlflow:
        with mlflow.start_run(run_name=f'Experiment - {model_name}'):
            train_ds = mlflow.data.from_pandas(pd.concat([X_train_split, y_train_split], axis=1), 
                                            targets=y_train_split.name, 
                                            name="nn_train_data")
            val_ds = mlflow.data.from_pandas(pd.concat([X_val_split, y_val_split], axis=1),
                                            targets=y_val_split.name,
                                            name="nn_val_data")
            test_ds = mlflow.data.from_pandas(pd.concat([X_test, y_test], axis=1),
                                            targets=y_test.name,
                                            name="nn_test_data")
            mlflow.log_input(train_ds, "nn_train_data")
            mlflow.log_input(val_ds, "nn_val_data")
            mlflow.log_input(test_ds, "nn_test_data")
            num_pos = torch.sum(y_train_tensor == 1).float()
            num_neg = torch.sum(y_train_tensor == 0).float()
            pos_weight = num_neg / (num_pos + 1e-8)
            pos_weight_tensor = torch.tensor([pos_weight])
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            input_dim = X_train.shape[1]
            model = NN(input_dim)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            best_model_state = None
            epochs_trained = 0
            for epoch in range(num_epochs):
                model.train()
                epoch_train_loss = 0.0
                num_batches = 0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                    num_batches += 1
                avg_train_loss = epoch_train_loss / num_batches
                train_losses.append(avg_train_loss)
                model.eval()
                epoch_val_loss = 0.0
                val_num_batches = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        val_outputs = model(X_batch)
                        val_loss = criterion(val_outputs, y_batch)
                        epoch_val_loss += val_loss.item()
                        val_num_batches += 1
                avg_val_loss = epoch_val_loss / val_num_batches
                val_losses.append(avg_val_loss)
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                if (epoch+1) % 5 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}')
                    model.load_state_dict(best_model_state)
                    epochs_trained = epoch + 1
                    break
                epochs_trained = epoch + 1
            print(f'Training completed. Total epochs trained: {epochs_trained}')
            model.eval()
            y_pred_prob_list = []
            with torch.no_grad():
                for X_batch, _ in test_loader:
                    batch_pred = model(X_batch).numpy()
                    y_pred_prob_list.append(batch_pred)
            y_pred_prob = np.vstack(y_pred_prob_list)
            y_pred = (y_pred_prob > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = auc(*roc_curve(y_test, y_pred_prob)[0:2])
            pr_auc = average_precision_score(y_test, y_pred_prob)
            print(f'\n{model_name} Performance:')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')
            print(f'ROC-AUC: {roc_auc:.4f}')
            print(f'PR-AUC: {pr_auc:.4f}')
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("patience", patience)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs_trained", epochs_trained)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("pr_auc", pr_auc)
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.pytorch.log_model(model, model_name)
            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/" + model_name, model_name)
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_val_loss': best_val_loss,
        'epochs_trained': epochs_trained,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }


nn_results = train_and_evaluate_nn(X_train_smote, y_train_smote, X_test, y_test, 
                                    model_name="Neural Network with SMOTE", NN=SimpleChurnNN)


class LogisticRegressionNN(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionNN, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


nn_results_logreg = train_and_evaluate_nn(X_train_smote, y_train_smote, X_test, y_test,
                                         model_name="Logistic Regression NN with SMOTE", NN=LogisticRegressionNN)


nn_results_logreg = train_and_evaluate_nn(X_train, y_train, X_test, y_test,
                                         model_name="Logistic Regression NN", NN=LogisticRegressionNN)


import nbformat
from nbformat import read, NO_CONVERT
def extract_code_from_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=NO_CONVERT)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    code_blocks = []
    for cell in code_cells:
        cell_lines = cell.source.splitlines()
        filtered_lines = [line for line in cell_lines if line.strip() and not line.strip().startswith('#')]
        if filtered_lines:
            code_blocks.append('\n'.join(filtered_lines))
    return '\n\n\n'.join(code_blocks)
code = extract_code_from_notebook('eda.ipynb')
with open('eda.py', 'w', encoding='utf-8') as f:
    f.write(code)