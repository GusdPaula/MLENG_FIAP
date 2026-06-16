"""Modelos de baseline para comparação do sistema de recomendação."""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression


class PopularityRecommender:
    """Recomendador baseline que pontua itens puramente com base em sua popularidade no conjunto de treino."""

    def __init__(self) -> None:
        self.item_popularity: dict[int, float] = {}
        self.default_score: float = 0.0

    def fit(self, interactions: pd.DataFrame) -> None:
        """Ajusta o modelo de popularidade nas interações de treino.

        Args:
            interactions: DataFrame contendo pelo menos a coluna 'item_idx', opcionalmente 'weight'.
        """
        # Soma os pesos ou conta ocorrências de cada item
        if "weight" in interactions.columns:
            pop = interactions.groupby("item_idx")["weight"].sum().to_dict()
        else:
            pop = interactions.groupby("item_idx").size().to_dict()

        # Normaliza os scores de popularidade para [0, 1] para facilitar a comparação
        if pop:
            max_val = max(pop.values())
            self.item_popularity = {k: v / max_val for k, v in pop.items()}
        else:
            self.item_popularity = {}

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Prediz scores de popularidade para pares de usuário-item.

        Args:
            users: Array com índices de usuários.
            items: Array com índices de itens.

        Returns:
            Array com scores de popularidade.
        """
        scores = np.array(
            [self.item_popularity.get(int(item), self.default_score) for item in items]
        )
        return scores


class LogisticRegressionRecommender:
    """Recomendador baseline que usa Logistic Regression sobre matrizes esparsas one-hot de usuário e item."""

    def __init__(self, num_users: int, num_items: int) -> None:
        self.num_users = num_users
        self.num_items = num_items
        self.model = LogisticRegression(max_iter=100, C=1.0, solver="liblinear")

    def _to_sparse_matrix(self, users: np.ndarray, items: np.ndarray) -> coo_matrix:
        """Converte índices de usuário e item em uma matriz esparsa codificada em one-hot."""
        num_samples = len(users)
        rows = np.repeat(np.arange(num_samples), 2)
        cols = np.column_stack([users, self.num_users + items]).ravel()
        data = np.ones(num_samples * 2)
        return coo_matrix(
            (data, (rows, cols)), shape=(num_samples, self.num_users + self.num_items)
        )

    def fit(self, users: np.ndarray, items: np.ndarray, labels: np.ndarray) -> None:
        """Ajusta o modelo de Logistic Regression nas amostras de treino.

        Args:
            users: Array com índices de usuários.
            items: Array com índices de itens.
            labels: Rótulos binários (0.0 ou 1.0).
        """
        X = self._to_sparse_matrix(users, items)
        self.model.fit(X, labels)

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Prediz a probabilidade de interação para pares de usuário-item.

        Args:
            users: Array com índices de usuários.
            items: Array com índices de itens.

        Returns:
            Array com probabilidades de interação.
        """
        X = self._to_sparse_matrix(users, items)
        # Retorna a probabilidade da classe 1
        return self.model.predict_proba(X)[:, 1]
