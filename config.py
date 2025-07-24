import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    """Unified configuration class."""
    cosmos_endpoint: str
    cosmos_key: str
    cosmos_database: str = "metadata"
    cosmos_container: str = "table_metadata"
    
    databricks_host: str = ""
    databricks_token: str = ""
    databricks_http_path: str = ""
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment."""
        load_dotenv("credentials.env")
        
        endpoint = os.getenv("COSMOS_DB_ENDPOINT")
        key = os.getenv("COSMOS_DB_KEY")
        
        if not endpoint or not key:
            raise ValueError("COSMOS_DB_ENDPOINT and COSMOS_DB_KEY must be set")
        
        return cls(
            cosmos_endpoint=endpoint,
            cosmos_key=key,
            cosmos_database=os.getenv("COSMOS_DB_DATABASE", "metadata"),
            cosmos_container=os.getenv("COSMOS_DB_CONTAINER", "table_metadata"),
            databricks_host=os.getenv("DATABRICKS_HOST", ""),
            databricks_token=os.getenv("DATABRICKS_TOKEN", ""),
            databricks_http_path=os.getenv("DATABRICKS_HTTP_PATH", "")
        )


def setup_logging() -> logging.Logger:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_contract_generator.log')
        ]
    )
    return logging.getLogger(__name__)
