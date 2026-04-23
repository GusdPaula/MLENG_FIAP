from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Pydantic will automatically look for these in your .env file
    google_cloud_project: str = Field(..., alias="GOOGLE_CLOUD_PROJECT")
    bq_dataset_id: str = "ibmchurn"
    bq_table_id: str = "ibm_churn"

    # Tells Pydantic to read from the .env file
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Create a single instance of the settings to be used across the application
settings = Settings()
