"""Pipeline sklearn reprodutível para Telco Churn."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
import pickle
from pathlib import Path
from typing import Optional


class TelcoPipeline:
    """Pipeline reprodutível para modelos Telco Churn."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipeline = None
        self.model_name = None
    
    def create_logistic_regression(self, 
                                   class_weight: str = 'balanced',
                                   max_iter: int = 1000) -> Pipeline:
        """Cria pipeline com LogisticRegression."""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])
        self.pipeline = pipeline
        self.model_name = 'LogisticRegression'
        return pipeline
    
    def create_random_forest(self,
                            n_estimators: int = 100,
                            max_depth: Optional[int] = 20,
                            min_samples_split: int = 10) -> Pipeline:
        """Cria pipeline com RandomForestClassifier."""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ))
        ])
        self.pipeline = pipeline
        self.model_name = 'RandomForest'
        return pipeline
    
    def create_xgboost(self,
                      n_estimators: int = 100,
                      max_depth: int = 5,
                      learning_rate: float = 0.1) -> Pipeline:
        """Cria pipeline com XGBoostClassifier."""
        # Calcular scale_pos_weight para lidar com desbalanceamento
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1,
                verbosity=0
            ))
        ])
        self.pipeline = pipeline
        self.model_name = 'XGBoost'
        return pipeline
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'TelcoPipeline':
        """Treina o pipeline."""
        if self.pipeline is None:
            raise ValueError("Pipeline não foi criado. Use create_* antes.")
        
        self.pipeline.fit(X_train, y_train)
        print(f"✅ {self.model_name} treinado com {X_train.shape[0]} amostras")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições."""
        if self.pipeline is None:
            raise ValueError("Pipeline não foi treinado.")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades."""
        if self.pipeline is None:
            raise ValueError("Pipeline não foi treinado.")
        
        if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
            return self.pipeline.named_steps['classifier'].predict_proba(X)
        else:
            # Para modelos que não têm predict_proba (ex: LinearSVM)
            raise AttributeError(f"{self.model_name} não suporta predict_proba")
    
    def save(self, filepath: str) -> None:
        """Salva pipeline em arquivo."""
        if self.pipeline is None:
            raise ValueError("Pipeline não foi treinado.")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"✅ Pipeline salvo em {filepath}")
    
    def load(self, filepath: str) -> 'TelcoPipeline':
        """Carrega pipeline de arquivo."""
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)
        print(f"✅ Pipeline carregado de {filepath}")
        return self
    
    @staticmethod
    def from_dict(config: dict) -> 'TelcoPipeline':
        """Cria pipeline a partir de config dict."""
        pipeline_obj = TelcoPipeline(random_state=config.get('random_state', 42))
        model_type = config.get('model_type')
        
        if model_type == 'logistic_regression':
            pipeline_obj.create_logistic_regression(
                class_weight=config.get('class_weight', 'balanced'),
                max_iter=config.get('max_iter', 1000)
            )
        elif model_type == 'random_forest':
            pipeline_obj.create_random_forest(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 20),
                min_samples_split=config.get('min_samples_split', 10)
            )
        elif model_type == 'xgboost':
            pipeline_obj.create_xgboost(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 5),
                learning_rate=config.get('learning_rate', 0.1)
            )
        else:
            raise ValueError(f"Modelo desconhecido: {model_type}")
        
        return pipeline_obj
