from typing import List, Optional

import psycopg2
import psycopg2.errors
from langchain_core.documents import Document
from langchain_community.vectorstores import PGVector
from langchain_text_splitters import MarkdownTextSplitter
from langchain_ollama import OllamaEmbeddings

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreConfig:
    """
    Configuration and management for the PGVector vector store.
    """

    def __init__(
        self,
        embeddings: Optional[OllamaEmbeddings],
        connection: str,
        collection_name: str,
    ) -> None:
        """
        Initialize the VectorStoreConfig.

        Args:
            embeddings (Optional[OllamaEmbeddings]): The embeddings model to use.
            connection (str): The database connection string.
            collection_name (str): The name of the collection in the vector store.
        """
        self.embeddings = embeddings
        self.connection = connection
        self.collection_name = collection_name

    def get_or_create(self) -> PGVector:
        """
        Get an existing vector store connection or create a new one.

        Returns:
            PGVector: The vector store instance.
        """
        vector_store = self.get_connection()
        if vector_store is None:
            vector_store = self.create_vector_store()
        return vector_store

    def get_connection(self) -> Optional[PGVector]:
        """
        Attempt to connect to an existing vector store index.

        Returns:
            Optional[PGVector]: The vector store instance if successful, None otherwise.
        """
        try:
            vector_store = PGVector.from_existing_index(
                embedding=self.embeddings,
                collection_name=self.collection_name,
                connection_string=self.connection,
                use_jsonb=True,
            )
            return vector_store
        except Exception as e:
            logger.warning(
                f"Could not connect to existing index (it may not exist yet): {e}"
            )
            return None

    def create_vector_store(self) -> PGVector:
        """
        Create a new vector store instance.

        Returns:
            PGVector: The newly created vector store instance.
        """
        vector_store = PGVector(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_string=self.connection,
            use_jsonb=True,
        )
        return vector_store


def clear_embedding(file_id: str) -> int:
    """
    Clear existing embeddings for a specific file ID from the database.

    Args:
        file_id (str): The ID of the file whose embeddings should be cleared.

    Returns:
        int: The number of deleted chunks.
    """
    connection_string = settings.database_url

    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
    except Exception as e:
        logger.exception("Failed to connect to the database to clear embeddings.")
        raise

    try:
        delete_query = """
            DELETE FROM langchain_pg_embedding
            WHERE cmetadata->>'file_id' = %s
        """

        cursor.execute(delete_query, (str(file_id),))
        deleted_count = cursor.rowcount
        conn.commit()

        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} chunks for file_id: {file_id}")
        else:
            logger.info(f"No existing embeddings found for file_id: {file_id}")

        return deleted_count

    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        logger.info(
            "Table 'langchain_pg_embedding' does not exist yet. Skipping clear."
        )
        return 0
    except Exception as e:
        conn.rollback()
        logger.exception(f"Error clearing embeddings for file_id {file_id}")
        raise
    finally:
        cursor.close()
        conn.close()


def embed_file(
    file_id: str,
    file_name: str,
    docs: List[Document],
) -> bool:
    """
    Split documents and embed them into the vector store.

    Args:
        file_id (str): The unique identifier for the file.
        file_name (str): The name of the file.
        docs (List[Document]): A list of Document objects to embed.

    Returns:
        bool: True if embedding was successful and chunks were added, False otherwise.
    """
    # Clears existing embedding for the same file id
    clear_embedding(file_id)

    child_splitter = MarkdownTextSplitter(chunk_size=3000, chunk_overlap=500)
    all_docs = []
    for doc in docs:
        sub_docs = child_splitter.split_documents([doc])
        for sub_doc in sub_docs:
            all_docs.append(sub_doc)

    # Embed all chunks directly
    if all_docs:
        try:
            embeddings = OllamaEmbeddings(model=settings.embedding_model)
            PGVector.from_documents(
                embedding=embeddings,
                documents=all_docs,
                collection_name=settings.collection_name,
                connection_string=settings.database_url,
                use_jsonb=True,
            )
            logger.info(
                f"Embedded {len(all_docs)} chunks from {file_name} into vector store"
            )
            return True
        except Exception as e:
            logger.exception(f"Failed to embed documents for {file_name}")
            return False
    else:
        logger.warning(f"No valid chunks to embed for file: {file_name}")
        return False
