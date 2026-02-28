import asyncio
import gc
import os
import re
import traceback
from typing import Optional, List, Dict, Any

import PyPDF2
import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    granite_picture_description,
    TableFormerMode,
    EasyOcrOptions,
    TableStructureOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

from src.config.settings import settings
from src.utils.logger import get_logger
from src.processing.image_processor import (
    extract_context_around_image,
    extract_images_from_markdown,
    process_images_async,
    replace_images_with_descriptions,
)

logger = get_logger(__name__)


def split_pdf(input_path: str, pages_per_split: int = 10) -> List[Dict[str, Any]]:
    """
    Split a PDF into smaller chunks.

    Args:
        input_path (str): The path to the input PDF file.
        pages_per_split (int, optional): Number of pages per split. Defaults to 10.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing split file info.
    """
    os.makedirs(settings.temporary_folder, exist_ok=True)
    split_files = []

    with open(input_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)

        for i in range(0, total_pages, pages_per_split):
            pdf_writer = PyPDF2.PdfWriter()

            # Add pages to current split
            end_page = min(i + pages_per_split, total_pages)
            for page_num in range(i, end_page):
                pdf_writer.add_page(pdf_reader.pages[page_num])

            # Save split PDF
            output_filename = f"split_{i//pages_per_split + 1}.pdf"
            output_path = os.path.join(settings.temporary_folder, output_filename)

            with open(output_path, "wb") as output_file:
                pdf_writer.write(output_file)

            split_files.append(
                {
                    "file": output_path,
                    "start_page": i + 1,  # 1-indexed
                    "end_page": end_page,
                }
            )

    return split_files


def combine_markdown_files(markdown_parts: List[Dict[str, Any]]) -> str:
    """
    Combine multiple markdown parts into one.

    Args:
        markdown_parts (List[Dict[str, Any]]): List of markdown parts.

    Returns:
        str: The combined markdown string.
    """
    combined_markdown = ""

    for part in markdown_parts:
        if part.get("markdown"):
            combined_markdown += part["markdown"]
            combined_markdown += "\n\n"
            combined_markdown += "<!-- page break -->"
            if not combined_markdown.endswith("\n"):
                combined_markdown += "\n\n"

    return combined_markdown


def split_markdown_by_pages(markdown_text: str) -> List[Dict[str, Any]]:
    """
    Split combined markdown text into per-page sections based on {N} page markers.

    The combined markdown uses ``{N}`` as page-boundary markers (produced by
    :func:`replace_page_breaks_with_numbers`).  Content *before* the first
    marker belongs to page 1, content *between* marker ``{k}`` and ``{k+1}``
    belongs to page ``k+1``, and so on.

    Args:
        markdown_text (str): The combined markdown with ``{N}`` page markers.

    Returns:
        List[Dict[str, Any]]: Ordered list of dicts, each with keys
        ``'page_number'`` (int, 1-indexed) and ``'markdown'`` (str).
    """
    # Split on {N} markers while capturing the numeric group
    parts = re.split(r"\{(\d+)\}", markdown_text)

    pages: List[Dict[str, Any]] = []
    # parts layout after split:
    #   parts[0]            – content before {1}  → page 1
    #   parts[1], parts[2]  – "1", content after {1} → page 2
    #   parts[3], parts[4]  – "2", content after {2} → page 3
    #   …
    # Even-indexed entries are page content; odd-indexed are the marker digits.
    for i, part in enumerate(parts):
        if i % 2 == 0:
            content = part.strip()
            if content:
                page_number = (i // 2) + 1
                pages.append({"page_number": page_number, "markdown": content})

    return pages


def replace_page_breaks_with_numbers(markdown_text: str, start_page: int = 1) -> str:
    """
    Replace page break placeholders with actual page numbers.

    Args:
        markdown_text (str): The markdown text containing page breaks.
        start_page (int, optional): The starting page number. Defaults to 1.

    Returns:
        str: The updated markdown text.
    """
    page_break_pattern = r"<!-- page break -->"

    current_page = start_page

    def replace_page_break(match: re.Match) -> str:
        nonlocal current_page
        replacement = f"{{{current_page}}}"
        current_page += 1
        return replacement

    # Replace page breaks with page numbers
    updated_markdown = re.sub(page_break_pattern, replace_page_break, markdown_text)

    return updated_markdown


def get_pdf_converter() -> Optional[DocumentConverter]:
    """
    Create and configure a DocumentConverter for PDF to markdown conversion.

    Returns:
        Optional[DocumentConverter]: The configured converter, or None if creation fails.
    """
    accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.AUTO
    )  # if cuda or mps is available it will automatically use it, otherwise it will fall back to CPU
    table_former_mode = TableFormerMode.FAST
    do_cell_matching = False
    generate_picture_images = True
    do_picture_classification = True
    do_picture_description = True

    ocr_options = EasyOcrOptions(
        lang=["en"],
        force_full_page_ocr=True,
    )

    table_structure_options = TableStructureOptions(
        do_cell_matching=do_cell_matching,
        mode=table_former_mode,
    )

    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,
        generate_picture_images=generate_picture_images,
        do_picture_classification=do_picture_classification,
        images_scale=1.0,
        do_picture_description=do_picture_description,
        picture_description_options=granite_picture_description,
        ocr_options=ocr_options,
        table_structure_options=table_structure_options,
        accelerator_options=accelerator_options,
    )
    try:
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        return converter
    except Exception as e:
        logger.exception("Error creating PDF converter")
        return None


def process_pdf(
    file_path: str,
    pages_per_split: int = 1,
    context_lines_before: int = 5,
    context_lines_after: int = 5,
) -> List[Dict[str, Any]]:
    """
    Process a PDF file, convert it to markdown, and process embedded images.

    Args:
        file_path (str): The path to the PDF file.
        pages_per_split (int, optional): Number of pages per split. Defaults to 1.
        context_lines_before (int, optional): Lines of context before an image. Defaults to 5.
        context_lines_after (int, optional): Lines of context after an image. Defaults to 5.

    Returns:
        List[Dict[str, Any]]: Ordered list of per-page dicts, each with keys
        ``'page_number'`` (int, 1-indexed) and ``'markdown'`` (str).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input PDF file '{file_path}' does not exist")

    split_files = split_pdf(file_path, pages_per_split)
    markdown_parts = []

    for i, split_info in enumerate(split_files):
        try:
            logger.info(
                f"Processing split {i+1}/{len(split_files)}: {split_info['file']}"
            )

            # Clear GPU cache before processing each split to manage memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            pdf_converter = None

            try:
                pdf_converter = get_pdf_converter()
                if pdf_converter is None:
                    logger.error(f"Failed to initialize PDF converter for split {i+1}")
                    continue

                result = pdf_converter.convert(split_info["file"])
            except Exception as e:
                logger.exception(f"Error converting split {i+1}")
                result = None

            if result is None or not hasattr(result, "document"):
                logger.error(f"Conversion result is invalid for split {i+1}")
                continue

            markdown_text = result.document.export_to_markdown(
                page_break_placeholder="<!-- page break -->",
                image_mode="embedded",
            )
            if markdown_text:
                markdown_parts.append(
                    {
                        "markdown": markdown_text,
                        "start_page": split_info["start_page"],
                        "end_page": split_info["end_page"],
                    }
                )

                logger.info(f"Successfully converted split {i+1}")

        except Exception as e:
            logger.exception(f"Error processing split {i+1}")
            continue
        finally:
            # Clear converter instance to free memory
            if pdf_converter is not None:
                del pdf_converter

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Combine markdown parts after processing all splits with page breaks
    combined_markdown = combine_markdown_files(markdown_parts)

    markdown_with_page_numbers = replace_page_breaks_with_numbers(
        combined_markdown, start_page=1
    )

    extracted_images = extract_images_from_markdown(markdown_with_page_numbers)

    if not extracted_images:
        logger.info("No images found in document")
        # Cleanup temporary files
        logger.info("Cleaning up temporary files...")
        for split_info in split_files:
            try:
                os.remove(split_info["file"])
            except OSError:
                pass

        return split_markdown_by_pages(markdown_with_page_numbers)

    contexts = []

    for i, image_info in enumerate(extracted_images, 1):
        # Extract context around this image
        context = extract_context_around_image(
            markdown_with_page_numbers,
            i,  # image ID (1-indexed)
            lines_before=context_lines_before,
            lines_after=context_lines_after,
        )
        contexts.append(context["combined"])
        logger.info(f"Extracted context for image {i}")

    # Run async image processing once for all images
    asyncio.run(process_images_async(extracted_images, contexts))

    final_markdown = replace_images_with_descriptions(
        markdown_with_page_numbers, extracted_images
    )

    for split_info in split_files:
        try:
            os.remove(split_info["file"])
        except OSError:
            pass

    return split_markdown_by_pages(final_markdown)
