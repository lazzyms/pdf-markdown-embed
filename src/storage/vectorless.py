from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pydantic import BaseModel

from src.models.llm_factory import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Tree(BaseModel):
    """
    A simple tree structure to represent the hierarchical organization of document sections.
    """

    node_id: str
    title: str
    content: str
    summary: str
    children: List["Tree"] = []


# Maps header marker → (metadata key, integer depth level)
_HEADER_MAP: List[tuple] = [
    ("#", "h1", 1),
    ("##", "h2", 2),
    ("###", "h3", 3),
    ("####", "h4", 4),
]

_HEADERS_TO_SPLIT_ON = [(marker, key) for marker, key, _ in _HEADER_MAP]
_LEVEL_FOR_KEY: Dict[str, int] = {key: level for _, key, level in _HEADER_MAP}


def get_tree(
    file_id: str,
    file_name: str,
    docs: List[Document],
) -> Tree:
    """
    Split documents by markdown headers and build a hierarchical Tree.

    The root node represents the whole document. Each # heading becomes a
    level-1 child, ## headings nest under the nearest preceding #, and so on
    down to ####.

    Args:
        file_id (str): The unique identifier for the file.
        file_name (str): The name of the file.
        docs (List[Document]): A list of Document objects (markdown content).

    Returns:
        Tree: The root node of the constructed document tree.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=True,
    )

    # Combine all pages/docs into a single markdown string so that headings
    # that span across page boundaries are handled correctly.
    full_markdown = "\n\n".join(doc.page_content for doc in docs)
    chunks = splitter.split_text(full_markdown)

    # Root node represents the entire document.
    root = Tree(
        node_id=f"{file_id}_root",
        title=file_name,
        content="",
        summary="",
        children=[],
    )

    # current_nodes[depth] holds the most-recently-seen node at that depth.
    # depth 0 is the root.
    current_nodes: Dict[int, Tree] = {0: root}

    for chunk_idx, chunk in enumerate(chunks):
        metadata = chunk.metadata  # e.g. {"h1": "Intro", "h2": "Background"}

        # Determine the deepest header level present in this chunk's metadata.
        depth = 0
        title = file_name
        for key, level in _LEVEL_FOR_KEY.items():
            if key in metadata and level > depth:
                depth = level
                title = metadata[key]

        # Chunks that precede any heading go directly under the root.
        if depth == 0:
            parent = root
        else:
            # Walk up until a parent at depth-1 is found.
            parent = _find_parent(current_nodes, depth)

        node = Tree(
            node_id=f"{file_id}_{chunk_idx}",
            title=title,
            content=chunk.page_content.strip(),
            summary="",
            children=[],
        )

        parent.children.append(node)

        # Register this node as the current node for its depth and
        # invalidate any deeper nodes that are no longer reachable.
        current_nodes[depth] = node
        for deeper in [d for d in current_nodes if d > depth]:
            del current_nodes[deeper]

    return root


def _find_parent(current_nodes: Dict[int, Tree], depth: int) -> Tree:
    """Return the nearest ancestor node for the given depth."""
    for candidate_depth in range(depth - 1, -1, -1):
        if candidate_depth in current_nodes:
            return current_nodes[candidate_depth]
    # Fallback (should never happen if root is always at 0).
    return current_nodes[0]


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a precise document analyst. "
                "Your task is to write a concise summary of a document section. "
                "Focus on key points, facts, and concepts. "
                "Keep the summary to 2-4 sentences. "
                "Do not include any preamble — output only the summary text."
            ),
        ),
        (
            "human",
            (
                "Section title: {title}\n\n"
                "Section content:\n{content}\n\n"
                "{child_context}"
                "Write a concise summary of this section."
            ),
        ),
    ]
)


def _build_child_context(node: Tree) -> str:
    """Build a text block listing child summaries for use in the parent prompt."""
    summaries = [
        f"- {child.title}: {child.summary}" for child in node.children if child.summary
    ]
    if not summaries:
        return ""
    return "Summaries of sub-sections:\n" + "\n".join(summaries) + "\n\n"


def _summarize_node(node: Tree, chain) -> None:
    """Recursively summarize a node bottom-up (children before parent)."""
    for child in node.children:
        _summarize_node(child, chain)

    content = node.content.strip() or "(No direct content — see sub-sections below.)"
    child_context = _build_child_context(node)

    logger.debug("Summarizing node: %r", node.title)
    response = chain.invoke(
        {
            "title": node.title,
            "content": content,
            "child_context": child_context,
        }
    )
    node.summary = str(response.content).strip()


def summarize_tree(root: Tree) -> Tree:
    """
    Walk the tree bottom-up and fill the ``summary`` field of every node
    using an LLM.  Children are summarized before their parent so that each
    parent prompt can include the already-computed child summaries for a
    richer, rolled-up view.

    Args:
        root (Tree): The root node returned by :func:`get_tree`.

    Returns:
        Tree: The same root node with all ``summary`` fields populated.
    """
    llm = get_llm_client()
    chain = _SUMMARY_PROMPT | llm

    # Summarize every child subtree first.
    for child in root.children:
        _summarize_node(child, chain)

    # Summarize the root itself using the rolled-up child context.
    child_context = _build_child_context(root)
    content = root.content.strip() or "(See sub-sections below.)"
    logger.debug("Summarizing root node: %r", root.title)
    response = chain.invoke(
        {
            "title": root.title,
            "content": content,
            "child_context": child_context,
        }
    )
    summary: str = str(response.content)
    root.summary = summary.strip()

    return root


def _summarize_leaf_node(node: Tree, chain) -> None:
    """
    Recursively walk the tree and summarize only leaf nodes (nodes without
    children).  Non-leaf nodes have their ``summary`` left as an empty string.
    """
    if not node.children:
        # This is a leaf — summarize it.
        content = node.content.strip() or "(No direct content.)"
        logger.debug("Summarizing leaf node: %r", node.title)
        response = chain.invoke(
            {
                "title": node.title,
                "content": content,
                "child_context": "",
            }
        )
        node.summary = str(response.content).strip()
    else:
        # Not a leaf — recurse into children, leave own summary empty.
        for child in node.children:
            _summarize_leaf_node(child, chain)


def summarize_leaves(root: Tree) -> Tree:
    """
    Walk the tree and fill the ``summary`` field **only for leaf nodes**
    (nodes that have no children) using an LLM.  Interior nodes and the root
    keep ``summary = ""``.  This minimises LLM calls while still capturing
    the finest-grained content in every branch.

    Args:
        root (Tree): The root node returned by :func:`get_tree`.

    Returns:
        Tree: The same root node with leaf ``summary`` fields populated.
    """
    llm = get_llm_client()
    chain = _SUMMARY_PROMPT | llm

    for child in root.children:
        _summarize_leaf_node(child, chain)

    return root
