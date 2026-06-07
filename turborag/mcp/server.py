"""
TurboRag MCP Server
===================
Exposes TurboRag as a Model Context Protocol (MCP) server so any
MCP-compatible agent (LangChain, Claude, CrewAI, etc.) can call it
as a tool set.

Tools exposed
-------------
  turborag_search(query, k)        — semantic search, returns JSON hits
  turborag_ask(question, k)        — full RAG answer
  turborag_index(text, metadata)   — add a document
  turborag_delete(doc_id)          — remove a document
  turborag_stats()                  — index statistics

Run::

    python -m turborag.mcp.server
    # or use as stdio MCP server:
    python -m turborag.mcp.server --transport stdio
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def create_mcp_server(rag=None):
    """Build and return a FastMCP server instance.

    Parameters
    ----------
    rag : TurboRag, optional
        Shared TurboRag instance.  Created lazily if not supplied.
    """
    try:
        from fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "fastmcp is required for the MCP server: pip install fastmcp"
        )

    from ..config import TurboRagConfig
    from ..core import TurboRag

    mcp = FastMCP("TurboRag")
    _state: Dict[str, Any] = {"rag": rag}

    def get_rag() -> TurboRag:
        if _state["rag"] is None:
            _state["rag"] = TurboRag()
        return _state["rag"]

    # ------------------------------------------------------------------
    # Tool: search
    # ------------------------------------------------------------------

    @mcp.tool()
    def turborag_search(query: str, k: int = 5) -> str:
        """Semantic search over the TurboRag knowledge base.

        Parameters
        ----------
        query : str   The natural-language search query.
        k     : int   Number of results to return (default 5, max 20).

        Returns JSON: {"results": [{"id", "text", "score", "metadata"}, ...]}
        """
        k = min(int(k), 20)
        hits = get_rag().search(query, k=k)
        results = [
            {"id": h.id, "text": h.text, "score": h.score, "metadata": h.metadata}
            for h in hits
        ]
        return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Tool: ask
    # ------------------------------------------------------------------

    @mcp.tool()
    def turborag_ask(question: str, k: int = 5) -> str:
        """Retrieve context from TurboRag and generate an answer with the local LLM.

        Parameters
        ----------
        question : str   The question to answer.
        k        : int   Number of context chunks to retrieve.

        Returns JSON: {"answer", "sources": [...]}
        """
        try:
            answer, sources = get_rag().ask(question, k=k)
            src = [{"id": s.id, "text": s.text[:200], "score": s.score} for s in sources]
            return json.dumps({"answer": answer, "sources": src}, ensure_ascii=False)
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

    # ------------------------------------------------------------------
    # Tool: index
    # ------------------------------------------------------------------

    @mcp.tool()
    def turborag_index(text: str, metadata: Optional[str] = None) -> str:
        """Add a document to the TurboRag knowledge base.

        Parameters
        ----------
        text     : str   Document text to index.
        metadata : str   Optional JSON string with metadata key-value pairs.

        Returns JSON: {"id", "status"}
        """
        meta = {}
        if metadata:
            try:
                meta = json.loads(metadata)
            except Exception:
                meta = {"raw": metadata}
        doc_id = get_rag().add_document(text, meta)
        return json.dumps({"id": doc_id, "status": "ok"})

    # ------------------------------------------------------------------
    # Tool: delete
    # ------------------------------------------------------------------

    @mcp.tool()
    def turborag_delete(doc_id: str) -> str:
        """Remove a document from the knowledge base by its ID.

        Parameters
        ----------
        doc_id : str   The document ID to delete.

        Returns JSON: {"id", "deleted": bool}
        """
        removed = get_rag().delete(doc_id)
        return json.dumps({"id": doc_id, "deleted": removed})

    # ------------------------------------------------------------------
    # Tool: stats
    # ------------------------------------------------------------------

    @mcp.tool()
    def turborag_stats() -> str:
        """Return statistics about the TurboRag index.

        Returns JSON with doc_count, dim, bit_width, model paths.
        """
        return json.dumps(get_rag().stats())

    return mcp


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="TurboRag MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="sse",
        help="MCP transport (default: sse)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    server = create_mcp_server()
    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport="sse", host=args.host, port=args.port)
