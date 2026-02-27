from minio import Minio
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MinioClient:
    """
    MinIO Python client for interacting with MinIO storage.
    """

    def __init__(self) -> None:
        """
        Initialize the MinIO client using settings.
        """
        if not settings.minio_root_user or not settings.minio_root_password:
            raise ValueError(
                "Missing MinIO credentials. Set MINIO_ROOT_USER and MINIO_ROOT_PASSWORD"
            )

        self.endpoint = settings.minio_endpoint
        self.bucket_name = settings.minio_default_bucket
        self._client = Minio(
            self.endpoint,
            access_key=settings.minio_root_user,
            secret_key=settings.minio_root_password,
            secure=False,
        )

    def ensure_bucket(self) -> None:
        """
        Ensure that the default bucket exists, creating it if necessary.
        """
        try:
            if not self._client.bucket_exists(self.bucket_name):
                self._client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket {self.bucket_name} already exists.")
        except Exception as e:
            logger.exception(f"Error ensuring bucket {self.bucket_name} exists.")
            raise

    def upload_file(
        self,
        object_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    ) -> None:
        """
        Upload a file to the MinIO bucket.

        Args:
            object_name (str): The name of the object in the bucket.
            file_path (str): The local path to the file to upload.
            content_type (str, optional): The content type of the file. Defaults to "application/octet-stream".
        """
        try:
            self._client.fput_object(
                self.bucket_name, object_name, file_path, content_type=content_type
            )
            logger.info(f"Uploaded {file_path} to {object_name}")
        except Exception as e:
            logger.exception(f"Error uploading {file_path} to {object_name}")
            raise

    def download_file(self, object_name: str, dest_path: str) -> None:
        """
        Download a file from the MinIO bucket.

        Args:
            object_name (str): The name of the object in the bucket.
            dest_path (str): The local path where the file should be saved.
        """
        try:
            self._client.fget_object(self.bucket_name, object_name, dest_path)
            logger.info(f"Downloaded {object_name} to {dest_path}")
        except Exception as e:
            logger.exception(f"Error downloading {object_name} to {dest_path}")
            raise

    def list_buckets(self) -> list:
        """
        List all buckets in the MinIO server.

        Returns:
            list: A list of bucket objects.
        """
        try:
            return self._client.list_buckets()
        except Exception as e:
            logger.exception("Error listing buckets")
            raise
