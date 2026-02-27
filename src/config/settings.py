import json
from typing import List, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    temporary_folder: str = "temp_files"
    collection_name: str = "default_collection"
    database_url: str = "postgresql://postgres:postgres@localhost:5433/postgres"
    files: str = "[]"
    embedding_model: str = "nomic-embed-text"

    # MinIO settings
    minio_endpoint: str = "localhost:9000"
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin"
    minio_default_bucket: str = "storage"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def files_list(self) -> List[Dict[str, Any]]:
        """
        Parse the FILES environment variable as JSON.

        Returns:
            List[Dict[str, Any]]: A list of file dictionaries.
        """
        try:
            return json.loads(self.files) if self.files else []
        except json.JSONDecodeError:
            return []


settings = Settings()
