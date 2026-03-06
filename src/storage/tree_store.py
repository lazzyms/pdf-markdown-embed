"""
Persistent storage for the VECTORLESS tree index.

Schema (auto-created on first use):

    tree_files
    ----------
    file_id   TEXT  PRIMARY KEY
    file_name TEXT  NOT NULL
    source    TEXT  NOT NULL          -- original file path / MinIO key
    created_at TIMESTAMPTZ DEFAULT now()

    tree_nodes
    ----------
    node_id    TEXT  PRIMARY KEY
    file_id    TEXT  REFERENCES tree_files(file_id) ON DELETE CASCADE
    parent_id  TEXT  REFERENCES tree_nodes(node_id) ON DELETE CASCADE  -- NULL for root
    title      TEXT  NOT NULL
    content    TEXT  NOT NULL DEFAULT ''
    summary    TEXT  NOT NULL DEFAULT ''
    depth      INT   NOT NULL          -- 0 = root, 1 = h1, 2 = h2 …
    position   INT   NOT NULL          -- sibling insertion order (0-based)
    created_at TIMESTAMPTZ DEFAULT now()
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config.settings import settings
from src.storage.vectorless import Tree
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DDL_FILES = """
CREATE TABLE IF NOT EXISTS tree_files (
    file_id    TEXT PRIMARY KEY,
    file_name  TEXT NOT NULL,
    source     TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_DDL_NODES = """
CREATE TABLE IF NOT EXISTS tree_nodes (
    node_id    TEXT PRIMARY KEY,
    file_id    TEXT NOT NULL REFERENCES tree_files(file_id) ON DELETE CASCADE,
    parent_id  TEXT REFERENCES tree_nodes(node_id) ON DELETE CASCADE,
    title      TEXT NOT NULL,
    content    TEXT NOT NULL DEFAULT '',
    summary    TEXT NOT NULL DEFAULT '',
    depth      INT  NOT NULL,
    position   INT  NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_DDL_IDX_FILE = """
CREATE INDEX IF NOT EXISTS idx_tree_nodes_file_id
    ON tree_nodes (file_id);
"""

_DDL_IDX_PARENT = """
CREATE INDEX IF NOT EXISTS idx_tree_nodes_parent_id
    ON tree_nodes (parent_id);
"""


def _ensure_tables(engine: Engine) -> None:
    """Create the ``tree_files`` and ``tree_nodes`` tables if they don't exist."""
    with engine.begin() as conn:
        conn.execute(text(_DDL_FILES))
        conn.execute(text(_DDL_NODES))
        conn.execute(text(_DDL_IDX_FILE))
        conn.execute(text(_DDL_IDX_PARENT))
    logger.debug("tree_files / tree_nodes tables ensured")


def _get_engine() -> Engine:
    """Return a SQLAlchemy engine from *settings.database_url*."""
    return create_engine(settings.database_url, pool_pre_ping=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clear_tree(file_id: str, engine: Optional[Engine] = None) -> None:
    """
    Remove all previously stored rows for *file_id*.

    Safe to call even when no rows exist.  The ``ON DELETE CASCADE`` on
    ``tree_nodes.file_id`` means deleting from ``tree_files`` also removes
    every associated node row.

    Args:
        file_id: The unique identifier for the file whose tree should be removed.
        engine:  Optional existing engine; a new one is created if not provided.
    """
    eng = engine or _get_engine()
    with eng.begin() as conn:
        conn.execute(
            text("DELETE FROM tree_files WHERE file_id = :fid"),
            {"fid": file_id},
        )
    logger.debug("Cleared existing tree rows for file_id=%r", file_id)


def store_tree(
    root: Tree,
    file_id: str,
    file_name: str,
    source: str,
    engine: Optional[Engine] = None,
) -> None:
    """
    Persist the in-memory *root* Tree to PostgreSQL.

    The traversal is breadth-first so that each parent row is inserted
    before its children (required to satisfy the ``parent_id`` FK).

    If a tree for *file_id* already exists it is **replaced**: ``clear_tree``
    is called first so re-processing a file is always idempotent.

    Args:
        root:      Root node returned by :func:`~src.storage.vectorless.get_tree`.
        file_id:   Unique identifier for the document.
        file_name: Human-readable file name (stored in ``tree_files``).
        source:    Original file path / MinIO object key.
        engine:    Optional existing SQLAlchemy engine.
    """
    eng = engine or _get_engine()
    _ensure_tables(eng)
    clear_tree(file_id, engine=eng)

    with eng.begin() as conn:
        # ------------------------------------------------------------------
        # 1. Insert the file metadata row.
        # ------------------------------------------------------------------
        conn.execute(
            text(
                """
                INSERT INTO tree_files (file_id, file_name, source)
                VALUES (:file_id, :file_name, :source)
                ON CONFLICT (file_id) DO UPDATE
                    SET file_name  = EXCLUDED.file_name,
                        source     = EXCLUDED.source,
                        created_at = now()
                """
            ),
            {"file_id": file_id, "file_name": file_name, "source": source},
        )

        # ------------------------------------------------------------------
        # 2. BFS over the tree — collect all (node, parent_id, depth, pos).
        # ------------------------------------------------------------------
        # Each queue entry: (node, parent_id_or_None, depth, position)
        queue: deque[tuple[Tree, Optional[str], int, int]] = deque()
        queue.append((root, None, 0, 0))

        node_rows = []
        while queue:
            node, parent_id, depth, position = queue.popleft()

            node_rows.append(
                {
                    "node_id": node.node_id,
                    "file_id": file_id,
                    "parent_id": parent_id,
                    "title": node.title,
                    "content": node.content,
                    "summary": node.summary,
                    "depth": depth,
                    "position": position,
                }
            )

            for pos, child in enumerate(node.children):
                queue.append((child, node.node_id, depth + 1, pos))

        # ------------------------------------------------------------------
        # 3. Bulk-insert all node rows (parent rows precede children because
        #    we used BFS).
        # ------------------------------------------------------------------
        conn.execute(
            text(
                """
                INSERT INTO tree_nodes
                    (node_id, file_id, parent_id, title, content, summary, depth, position)
                VALUES
                    (:node_id, :file_id, :parent_id, :title, :content, :summary, :depth, :position)
                """
            ),
            node_rows,
        )

    logger.info(
        "Stored tree for file_id=%r: %d nodes inserted",
        file_id,
        len(node_rows),
    )
