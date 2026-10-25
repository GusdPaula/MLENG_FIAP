import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_and_scale(
    df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42
):
    """Split the DataFrame into training and testing sets and scale the features."""
    logger.info("Starting split and scale...")

    # Split features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Select float columns only
    float_cols = X_train.select_dtypes(include=["float"]).columns

    # Scale using training data only
    scaler = StandardScaler()

    X_train.loc[:, float_cols] = scaler.fit_transform(X_train[float_cols])

    X_test.loc[:, float_cols] = scaler.transform(X_test[float_cols])

    cols_with_nan_test = X_test.columns[X_test.isna().any()]
    cols_with_nan_train = X_train.columns[X_train.isna().any()]

    # drop the rows with NaN values in the test set
    X_test = X_test.dropna(subset=cols_with_nan_test)
    y_test = y_test.loc[X_test.index]

    # drop the rows with NaN values in the train set
    X_train = X_train.dropna(subset=cols_with_nan_train)
    y_train = y_train.loc[X_train.index]

    feature_columns = X_train.columns.tolist()

    logger.info("Split and scale completed.")

    return X_train, X_test, y_train, y_test, scaler, feature_columns
