import asyncio
import base64
import re
from typing import Dict, List, Any

from src.models.llm_factory import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

custom_prompt = """
Provide a clear, detailed, and accurate description of the image from the TATA Power AGM report. The description should allow a reader to fully understand the image without seeing it.

Include:
1. Image Type: (e.g., photograph, chart, infographic, table, diagram).
2. Layout and Composition: Describe the structure, key elements, color scheme, and any labels, captions, legends, or highlights.
3. Content Details:
   • If a table: describe column headers, notable entries, and visible patterns or trends.
   • If a chart/graph: specify chart type, axis labels, legend items, and key insights.
   • If a diagram: describe shapes, flows, relationships, and labeled sections.
   • If a photograph: describe subjects, actions, setting, and visible branding.
4. Text Elements: Transcribe all visible text and describe where it appears and why (titles, labels, callouts, footnotes, etc.).
5. Visual Styling: Note colors, corporate branding, icons, and emphasis techniques.
6. Purpose and Context: Explain the likely message or intent of the image in the AGM report (e.g., financial performance, sustainability initiatives, strategic updates).

The description should be complete enough for someone to reconstruct the image accurately from your text.
"""


def extract_images_from_markdown(markdown_text: str) -> List[Dict[str, Any]]:
    """
    Extract images from markdown with embedded base64 images.

    Args:
        markdown_text (str): The markdown text containing embedded images.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing image information.
    """
    image_pattern = r"!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)"
    matches = re.findall(image_pattern, markdown_text)

    extracted_images = []

    for i, (alt_text, image_format, base64_data) in enumerate(matches, 1):
        try:
            # Decode base64 data for validation
            image_data = base64.b64decode(base64_data)

            image_info = {
                "id": i,
                "alt_text": alt_text,
                "format": image_format,
                "base64_data": base64_data,
                "file_size": len(image_data),
                "description": None,
            }

            extracted_images.append(image_info)
            logger.info(f"Found image {i}: {alt_text} ({len(image_data)} bytes)")

        except Exception as e:
            logger.exception(f"Error processing image {i}")
            continue

    return extracted_images


def extract_context_around_image(
    markdown_text: str, image_id: int, lines_before: int = 5, lines_after: int = 5
) -> Dict[str, str]:
    """
    Extract context (text lines) around an image reference in the markdown.

    Args:
        markdown_text (str): The full markdown text.
        image_id (int): The image ID/number (1-indexed).
        lines_before (int, optional): Number of lines to extract before the image. Defaults to 5.
        lines_after (int, optional): Number of lines to extract after the image. Defaults to 5.

    Returns:
        Dict[str, str]: A dictionary with 'before', 'after', and 'combined' context.
    """
    # Use the same pattern as extract_images_from_markdown to find base64 embedded images
    image_pattern = r"!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)"

    matches = list(re.finditer(image_pattern, markdown_text))

    if image_id > len(matches) or image_id < 1:
        return {"before": "", "after": "", "combined": ""}

    # Get the match for this specific image (1-indexed)
    match = matches[image_id - 1]
    image_pos = match.start()

    # Extract text before the image
    text_before_image = markdown_text[:image_pos]
    lines_before_list = text_before_image.split("\n")
    context_before = (
        "\n".join(lines_before_list[-lines_before:])
        if len(lines_before_list) >= lines_before
        else text_before_image
    )

    # Extract text after the image
    text_after_image = markdown_text[match.end() :]
    lines_after_list = text_after_image.split("\n")
    context_after = (
        "\n".join(lines_after_list[:lines_after])
        if len(lines_after_list) >= lines_after
        else text_after_image
    )

    # Combine context
    combined_context = (
        f"Text before image:\n{context_before}\n\nText after image:\n{context_after}"
    )

    return {
        "before": context_before.strip(),
        "after": context_after.strip(),
        "combined": combined_context.strip(),
    }


def create_prompt_with_context(base_prompt: str, context: str) -> str:
    """
    Creates an enhanced prompt that includes context from the document.

    Args:
        base_prompt (str): The base prompt for image description.
        context (str): Text context (before and after the image).

    Returns:
        str: Enhanced prompt string.
    """
    if context and context.strip():
        enhanced_prompt = f"""{base_prompt}

            **Additional Context from Document:**
            The image appears in the following context within the document:

            {context}

            Please use this surrounding text to provide more accurate and contextually relevant descriptions of the image, especially for technical diagrams, charts, or figures that may be referenced in the text."""
        return enhanced_prompt
    else:
        return base_prompt


async def describe_image_with_llm_async(
    base64_data: str, image_format: str, context: str = ""
) -> str:
    """
    Get detailed image description using LLM asynchronously.

    Args:
        base64_data (str): Base64 encoded image data.
        image_format (str): Image format (e.g., 'png', 'jpeg').
        context (str, optional): Optional context from surrounding text. Defaults to "".

    Returns:
        str: Image description string.
    """
    try:
        # Create enhanced prompt with context
        prompt_text = create_prompt_with_context(custom_prompt, context)

        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{base64_data}"
                    },
                },
            ],
        }

        llm_client = get_llm_client()
        # Use ainvoke for async operation
        response = await llm_client.ainvoke([message])
        llm_response = response.content
        return llm_response
    except Exception as e:
        logger.exception("Error describing image")
        return "Error generating description for image"


async def process_single_image(
    extracted_images: List[Dict[str, Any]],
    i: int,
    image_info: Dict[str, Any],
    context: str,
) -> str:
    """
    Process a single image to generate its description.

    Args:
        extracted_images (List[Dict[str, Any]]): The list of all extracted images.
        i (int): The index of the current image (1-based).
        image_info (Dict[str, Any]): The dictionary containing image data.
        context (str): The surrounding text context.

    Returns:
        str: The generated description.
    """
    logger.info(f"Getting description for image {i}/{len(extracted_images)}...")
    description = await describe_image_with_llm_async(
        image_info["base64_data"],
        image_info["format"],
        context=context,
    )
    image_info["description"] = description
    logger.info(f"✓ Description generated for image {i}")
    return description


async def process_images_async(
    extracted_images: List[Dict[str, Any]], contexts: List[str]
) -> None:
    """
    Process all images concurrently using asyncio.

    Args:
        extracted_images (List[Dict[str, Any]]): The list of extracted images.
        contexts (List[str]): The list of contexts corresponding to each image.
    """
    tasks = [
        process_single_image(extracted_images, i, image_info, contexts[i - 1])
        for i, image_info in enumerate(extracted_images, 1)
    ]

    # Run all tasks concurrently
    await asyncio.gather(*tasks)


def replace_images_with_descriptions(
    markdown_text: str, images: List[Dict[str, Any]]
) -> str:
    """
    Replace image patterns with LLM descriptions.

    Args:
        markdown_text (str): The markdown text containing embedded images.
        images (List[Dict[str, Any]]): List of image info dictionaries with descriptions.

    Returns:
        str: Updated markdown with images replaced by descriptions.
    """
    updated_markdown = markdown_text

    for image_info in images:
        # Find the original embedded image pattern
        pattern = rf'!\[([^\]]*)\]\(data:image/{image_info["format"]};base64,{re.escape(image_info["base64_data"])}\)'

        # Create replacement with description
        description = image_info.get("description", "No description available.")

        replacement = f"**[Image Description]**\n\n{description}"

        updated_markdown = re.sub(pattern, replacement, updated_markdown)

    return updated_markdown
