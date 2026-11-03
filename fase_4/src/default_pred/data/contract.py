import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

ALLOWED_TERMS = [" 36 months", " 60 months", "36 months", "60 months"]
ALLOWED_GRADES = ["A", "B", "C", "D", "E", "F", "G"]
MONTH_YEAR_PATTERN = r"^[A-Z][a-z]{2}-\d{4}$"


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

    annual_inc: Series[float] = pa.Field(
        gt=20_000,
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
