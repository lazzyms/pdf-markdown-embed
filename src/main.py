import os

from langchain_core.documents import Document

from src.config.settings import settings
from src.processing.document_processor import process_pdf
from src.storage.vector_store import embed_file
from src.storage.minio_client import MinioClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_embedding(file_path: str, file_id: str, file_name: str) -> bool:
    """
    Process a PDF file and generate embeddings for it.

    Args:
        file_path (str): The local path to the downloaded PDF file.
        file_id (str): The unique identifier for the file.
        file_name (str): The name of the file.

    Returns:
        bool: True if embedding was successful, False otherwise.
    """
    logger.info(f"Generating embedding for {file_name} (ID: {file_id}) at {file_path}")

    try:
        markdown_text = process_pdf(file_path)

        # Wrap the markdown text in a Document object as expected by embed_file
        docs = [
            Document(
                page_content=markdown_text,
                metadata={
                    "file_id": file_id,
                    "source": file_path,
                    "file_name": file_name,
                },
            )
        ]

        return embed_file(file_id, file_name, docs)
    except Exception as e:
        logger.exception(f"Failed to generate embedding for {file_name}")
        return False


def main() -> None:
    """
    Main entry point for the application.
    """
    logger.info("Starting MD Converter application...")

    try:
        # Create a client using environment or defaults
        client = MinioClient()
        client.ensure_bucket()

        buckets = client.list_buckets()
        logger.info(f"Connected to MinIO endpoint: {client.endpoint}")
        logger.info("Buckets:")
        for b in buckets:
            logger.info(f"- {b.name}")

        # Create the temporary folder if not exists
        if not os.path.exists(settings.temporary_folder):
            os.mkdir(settings.temporary_folder)

        files_list = settings.files_list
        logger.info(f"Processing {len(files_list)} files for embedding generation...")

        for file in files_list:
            file_path = file.get("path")
            file_id = file.get("id")
            file_name = file.get("name")

            if not all([file_path, file_id, file_name]):
                logger.warning(f"Skipping invalid file entry: {file}")
                continue

            download_file_path = os.path.join(
                settings.temporary_folder, os.path.basename(file_path)
            )

            try:
                client.download_file(
                    file_path,
                    download_file_path,
                )
                generate_embedding(download_file_path, file_id, file_name)
            except Exception as e:
                logger.error(f"Failed to process file {file_name}: {e}")

    except Exception as e:
        logger.exception("Application encountered a fatal error")


if __name__ == "__main__":
    main()
