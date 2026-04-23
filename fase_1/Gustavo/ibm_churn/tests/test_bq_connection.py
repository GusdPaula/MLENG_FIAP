# test bigquery connection
import dotenv
from google.cloud import bigquery


def test_bq_connection():
    # Load environment variables from .env file s
    dotenv.load_dotenv()

    # Get the project ID from environment variables
    project_id = "ibm-churn"  # You can also get this from an environment variable

    # Create a BigQuery client
    client = bigquery.Client(project=project_id)

    # Run a simple query to test the connection
    query = "SELECT 1 AS test"
    result = client.query(query).to_dataframe()

    msg = "BigQuery connection test failed: No results returned"
    # Check if the query returned the expected result
    assert result["test"][0] == 1, msg
