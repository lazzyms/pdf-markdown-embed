from langchain_ollama import ChatOllama
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_llm_client() -> ChatOllama:
    """
    Get an instance of the ChatOllama LLM client.

    Returns:
        ChatOllama: The configured LLM client.
    """
    try:
        # Note: If you need a separate vision model, you might want to add a VISION_MODEL to settings
        # and pass it here depending on the use case. For now, we use the embedding_model or a default.
        # Actually, for image description, a vision model like 'llava' is typically needed.
        # Let's use the embedding_model from settings, but ideally it should be a vision-capable model.
        model_name = settings.llm_model
        logger.info(f"Initializing ChatOllama with model: {model_name}")
        return ChatOllama(model=model_name, temperature=0.2)
    except Exception as e:
        logger.exception("Failed to initialize LLM client")
        raise
