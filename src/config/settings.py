import json
from typing import List, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    temporary_folder: str = os.getenv("TEMPORARY_FOLDER", "temp_files")
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/mydb"
    )
    files: str = os.getenv("FILES", "[]")  # Expecting a JSON string of file info
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3:latest")

    # MinIO settings
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    minio_root_user: str = os.getenv("MINIO_ROOT_USER", "minioadmin")
    minio_root_password: str = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
    minio_default_bucket: str = os.getenv("MINIO_DEFAULT_BUCKET", "storage")

    # Processing strategy: "embed" (vector) or "vectorless" (tree index)
    process_type: str = os.getenv("PROCESS_TYPE", "embed")

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
