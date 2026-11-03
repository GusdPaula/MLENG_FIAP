import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on the input DataFrame.

    Applies Privacy by Design (PII dropping), eliminates target leakage,
    removes geographic redlining proxies (Fairness F3 control), and
    calculates deterministic credit history duration at loan origination.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    logger.info("Starting feature engineering...")
    df = df.copy()

    # 1. Adjusting target
    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].map(
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

    # 2. Dropping PII and direct identifiers (Privacy by Design - LGPD Art. 6º, III)
    pii_columns = ["id", "member_id", "url", "title", "emp_title", "desc"]
    df = df.drop(columns=pii_columns, errors="ignore")

    # 3. Dropping post-origination / target leakage features
    # These features only occur post-issuance and cause target leakage in models
    post_origination_leakage_columns = [
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "last_credit_pull_d",
        "last_fico_range_high",
        "last_fico_range_low",
        "debt_settlement_flag",
        "debt_settlement_flag_date",
        "settlement_status",
        "settlement_date",
        "settlement_amount",
        "settlement_percentage",
        "settlement_term",
        "hardship_flag",
        "hardship_type",
        "hardship_reason",
        "hardship_status",
        "deferral_term",
        "hardship_amount",
        "hardship_start_date",
        "hardship_end_date",
        "payment_plan_start_date",
        "hardship_length",
        "hardship_dpd",
        "hardship_loan_status",
        "orig_projected_additional_accrued_interest",
        "hardship_payoff_balance_amount",
        "hardship_last_payment_amount",
        "pymnt_plan",
    ]
    df = df.drop(columns=post_origination_leakage_columns, errors="ignore")

    # 4. Dropping proxy for redlining / algorithmic fairness (Controle F3 LGPD)
    df = df.drop(columns=["zip_code"], errors="ignore")

    # 5. Dropping columns with missing values % > 20 (evaluated when sample > 10)
    if len(df) > 10:
        high_na_cols = df.columns[df.isna().mean() * 100 > 20]
        df = df.drop(columns=high_na_cols)

    # 6. Dropping categorical columns with high cardinality (> 20% unique values)
    cols_to_drop = [
        col
        for col in df.select_dtypes(include="object").columns
        if len(df) > 10 and (df[col].nunique() / len(df) * 100 > 20)
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # 7. Rank encoding
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

    # 8. Deterministic date encoding (Fixed reference at loan origination)
    if "issue_d" in df.columns and "earliest_cr_line" in df.columns:
        issue_dates = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        earliest_dates = pd.to_datetime(
            df["earliest_cr_line"], format="%b-%Y", errors="coerce"
        )
        diff_days = (issue_dates - earliest_dates).dt.days
        df["credit_history_months"] = (
            (diff_days // 30).fillna(0).clip(lower=0).astype(float)
        )

    # Drop raw date columns
    raw_date_cols = [
        "issue_d",
        "earliest_cr_line",
        "sec_app_earliest_cr_line",
    ]
    df = df.drop(columns=raw_date_cols, errors="ignore")

    # 9. One-hot encoding
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

    # 10. Drop rows with NaN in any float column (only when training on batches)
    if len(df) > 10:
        float_cols = df.select_dtypes(include="float").columns
        df = df.dropna(subset=float_cols)

    logger.info("Feature engineering completed.")
    return df
