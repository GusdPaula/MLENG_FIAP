import pandera.pandas as pa
from pandera.typing import Series


class Contract(pa.DataFrameModel):
    """
    Contract model for validating the input data.
    """

    annual_inc: Series[float] = pa.Field(
        gt=20_000,
        nullable=False,
    )

    purpose: Series[str] = pa.Field(
        nullable=False,
    )

    grade: Series[str] = pa.Field(
        isin=["A", "B", "C", "D", "E", "F", "G"],
        nullable=False,
    )
