import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on the input DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """

    logger.info("Starting feature engineering...")
    # Dropping column with missing values % > 20
    df = df.drop(columns=df.columns[df.isna().mean() * 100 > 20])

    # 1. Adjusting target
    df.loan_status = df.loan_status.map(
        {
            "Fully Paid": 0,
            "Charged Off": 1,
            "Default": 1,
            "Late (31-120 days)": 1,
            "Late (16-30 days)": 1,
            "In Grace Period": 1,
            "Current": 0,
            "Does not meet the credit policy. Status:Fully Paid": 0,
            "Does not meet the credit policy. Status:Charged Off": 1,
        }
    )

    # 2. Dropping ID and Title
    df = df.drop(columns=["id", "title"])

    # 3. Dropping categorical columns with more than 20% unique values
    cols_to_drop = [
        col
        for col in df.select_dtypes(include="object").columns
        if df[col].nunique() / len(df) * 100 > 20
    ]

    df = df.drop(columns=cols_to_drop)

    # 4. Rank encoding
    rank_columns = {
        "grade": list("ABCDEFG"),
        "sub_grade": [f"{grade}{i}" for grade in "ABCDEFG" for i in range(1, 6)],
        "emp_length": [
            "< 1 year",
            "1 year",
            "2 years",
            "3 years",
            "4 years",
            "5 years",
            "6 years",
            "7 years",
            "8 years",
            "9 years",
            "10+ years",
        ],
    }

    for col, categories in rank_columns.items():
        if col in df.columns:
            mapping = {value: rank for rank, value in enumerate(categories)}
            df[col] = df[col].map(mapping)

    # 5. Binary encoding
    binary_columns = [
        "pymnt_plan",
        "debt_settlement_flag",
        "hardship_flag",
    ]

    binary_mapping = {
        "n": 0,
        "y": 1,
        "no": 0,
        "yes": 1,
    }

    for col in binary_columns:
        if col in df.columns:
            df[col] = (
                df[col].astype("string").str.strip().str.lower().map(binary_mapping)
            )

    # 5. Date encoding
    date_columns = [
        "issue_d",
        "earliest_cr_line",
        "last_pymnt_d",
        "last_credit_pull_d",
        "next_pymnt_d",
        "sec_app_earliest_cr_line",
        "hardship_start_date",
        "hardship_end_date",
        "payment_plan_start_date",
        "debt_settlement_flag_date",
        "settlement_date",
    ]

    today = pd.Timestamp.today().normalize()

    for col in date_columns:
        if col in df.columns:
            dates = pd.to_datetime(df[col], format="%b-%Y", errors="coerce")

            df[f"{col}_days_from_today"] = (today - dates).dt.days

            df = df.drop(columns=col)

    # 7. ZIP code: extract digits
    if "zip_code" in df.columns:
        df["zip_code"] = (
            df["zip_code"].astype("string").str.extract(r"(\d+)", expand=False)
        )

        df["zip_code"] = pd.to_numeric(df["zip_code"], errors="coerce")

    # 8. One-hot encoding
    onehot_columns = [
        "term",
        "home_ownership",
        "verification_status",
        "purpose",
        "addr_state",
        "initial_list_status",
        "disbursement_method",
        "application_type",
    ]

    existing_columns = [col for col in onehot_columns if col in df.columns]

    df = pd.get_dummies(df, columns=existing_columns, dummy_na=False, dtype=int)

    # 10. Drop rows with NaN in any float column
    float_cols = df.select_dtypes(include="float").columns

    df = df.dropna(subset=float_cols)

    logger.info("Feature engineering completed.")

    return df
