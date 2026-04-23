from data.data_loader import BigQueryDataLoader


def test_data_loader_init():
    # Test that the loader initializes without crashing
    # (Note: This might need mock credentials to pass in CI)
    # Load data from BigQuery
    data_loader = BigQueryDataLoader()
    data_loader.load_data()
    df = data_loader.df

    # Check that the DataFrame is not empty
    assert df is not None, "DataFrame should not be None"
    assert not df.empty, "DataFrame should not be empty"


def test_data_loader_columns():
    # Test that the DataFrame has the expected columns
    data_loader = BigQueryDataLoader()
    data_loader.load_data()
    df = data_loader.df

    expected_columns = [
        "CustomerIDx",
        "Count",
        "Country",
        "State",
        "City",
        "Zip Code",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure Months",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charges",
        "Total Charges",
        "Churn Label",
        "Churn Value",
        "Churn Score",
        "CLTV",
        "Churn Reason",
    ]

    expected_columns = set(expected_columns)

    msg = f"DataFrame should contain columns: {expected_columns}"
    assert expected_columns.issubset(df.columns), msg
