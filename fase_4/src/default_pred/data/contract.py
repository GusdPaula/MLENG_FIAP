import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

ALLOWED_TERMS = [" 36 months", " 60 months", "36 months", "60 months"]
ALLOWED_GRADES = ["A", "B", "C", "D", "E", "F", "G"]
MONTH_YEAR_PATTERN = r"^[A-Z][a-z]{2}-\d{4}$"
MAX_PLAUSIBLE_INCOME = 10_000_000.0


def compute_effective_income(df: pd.DataFrame) -> pd.Series:
    """Compute effective annual income for Individual and Joint applications."""
    primary = pd.to_numeric(df["annual_inc"], errors="coerce").fillna(0.0)
    if "application_type" not in df.columns or "annual_inc_joint" not in df.columns:
        return primary

    app_type = df["application_type"].astype("string").str.strip()
    joint_inc = pd.to_numeric(df["annual_inc_joint"], errors="coerce")
    is_joint = app_type == "Joint App"
    joint_effective = pd.Series(
        np.where(joint_inc.notna(), np.maximum(primary, joint_inc.fillna(0.0)), 0.0),
        index=df.index,
        dtype="float64",
    )
    return pd.Series(
        np.where(is_joint, joint_effective, primary),
        index=df.index,
        dtype="float64",
    )


class Contract(pa.DataFrameModel):
    """Contract model for validating single and batch credit input payloads."""

    id: Series[int] = pa.Field(
        gt=0,
        unique=True,
        nullable=False,
        coerce=True,
    )

    loan_amnt: Series[float] = pa.Field(
        gt=0,
        le=100_000,
        nullable=False,
        coerce=True,
    )

    term: Series[str] = pa.Field(
        isin=ALLOWED_TERMS,
        nullable=False,
    )

    # Layer 1: Physical/scale domain check (blocks negative, null, or >10M typos)
    annual_inc: Series[float] = pa.Field(
        ge=0,
        le=MAX_PLAUSIBLE_INCOME,
        nullable=False,
        coerce=True,
    )

    purpose: Series[str] = pa.Field(
        nullable=False,
    )

    grade: Series[str] = pa.Field(
        isin=ALLOWED_GRADES,
        nullable=False,
    )

    issue_d: Series[str] = pa.Field(
        str_matches=MONTH_YEAR_PATTERN,
        nullable=False,
    )

    earliest_cr_line: Series[str] = pa.Field(
        str_matches=MONTH_YEAR_PATTERN,
        nullable=False,
    )

    @pa.dataframe_check(
        name="effective_income_positive_and_within_scale",
        error=(
            "effective annual income must be > 0 and <= 10,000,000 "
            "(annual_inc > 0 for Individual, or annual_inc_joint > 0 for Joint App)"
        ),
    )
    @classmethod
    def check_effective_income_positive(cls, df: pd.DataFrame) -> Series[bool]:
        """Layer 2: Ensure effective income is > 0 for Individual or Joint App."""
        primary = pd.to_numeric(df["annual_inc"], errors="coerce")
        effective = compute_effective_income(df)
        joint_valid = pd.Series(True, index=df.index)
        if "annual_inc_joint" in df.columns:
            joint_raw = pd.to_numeric(df["annual_inc_joint"], errors="coerce")
            joint_valid = joint_raw.isna() | (
                (joint_raw >= 0) & (joint_raw <= MAX_PLAUSIBLE_INCOME)
            )
        return (
            primary.notna()
            & (primary >= 0)
            & (effective > 0)
            & (effective <= MAX_PLAUSIBLE_INCOME)
            & joint_valid
        )

    @pa.dataframe_check(
        name="earliest_cr_line_not_after_issue_d",
        error="earliest_cr_line must be earlier than or equal to issue_d",
    )
    @classmethod
    def check_temporal_consistency(cls, df: pd.DataFrame) -> Series[bool]:
        """Ensure credit line opening date is not after the loan issue date."""
        issue_dates = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        earliest_dates = pd.to_datetime(
            df["earliest_cr_line"], format="%b-%Y", errors="coerce"
        )
        return (
            issue_dates.notna()
            & earliest_dates.notna()
            & (earliest_dates <= issue_dates)
        )
