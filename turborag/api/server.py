"""
TurboRag REST API
=================
FastAPI server exposing TurboRag as an HTTP service.

Endpoints
---------
  POST /index           — ingest a document
  POST /index/batch     — ingest many documents
  POST /search          — semantic search
  POST /ask             — RAG answer
  POST /embed           — get raw embedding
  DELETE /document/{id} — remove a document
  GET  /stats           — index statistics
  GET  /health          — liveness probe

Run with::

    python -m turborag.api.server
    # or
    uvicorn turborag.api.server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    text: str = Field(..., description="Document text to index")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk: bool = Field(False, description="Auto-chunk long text before indexing")


class IndexBatchRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    chunk: bool = False


class IndexResponse(BaseModel):
    id: str
    status: str = "ok"


class IndexBatchResponse(BaseModel):
    ids: List[str]
    count: int
    status: str = "ok"


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    k: int = Field(5, ge=1, le=100)
    filter_ids: Optional[List[str]] = None


class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    count: int


class AskRequest(BaseModel):
    question: str
    k: int = Field(5, ge=1, le=20)
    system: str = "You are a helpful assistant. Answer based on the provided context."


class AskResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    question: str


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: List[float]
    dim: int


class StatsResponse(BaseModel):
    doc_count: int
    dim: Optional[int]
    bit_width: int
    index_path: str
    embed_model: str
    llm_model: Optional[str]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(rag=None) -> FastAPI:
    """Create and return the FastAPI app.

    Parameters
    ----------
    rag : TurboRag, optional
        Pre-built TurboRag instance.  If None, one is created lazily from
        environment / default config on first request.
    """
    from ..config import TurboRagConfig

    app = FastAPI(
        title="TurboRag API",
        description="Local quantized RAG engine — low CPU, low RAM, fully offline.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow all by default (restrict in production via env)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Optional API key auth
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
    REQUIRED_KEY = os.getenv("TURBORAG_API_KEY", None)

    def verify_key(key: Optional[str] = Security(api_key_header)):
        if REQUIRED_KEY and key != REQUIRED_KEY:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or missing API key",
            )

    # Shared TurboRag instance (lazy init)
    _state: Dict[str, Any] = {"rag": rag}

    def get_rag():
        if _state["rag"] is None:
            from ..core import TurboRag
            _state["rag"] = TurboRag()
        return _state["rag"]

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/stats", response_model=StatsResponse)
    def stats():
        verify_key()
        return get_rag().stats()

    @app.post("/embed", response_model=EmbedResponse)
    def embed(req: EmbedRequest):
        verify_key()
        r = get_rag()
        vec = r.embedder.embed(req.text)
        return EmbedResponse(embedding=vec.tolist(), dim=len(vec))

    @app.post("/index", response_model=IndexResponse)
    def index_doc(req: IndexRequest):
        verify_key()
        r = get_rag()
        if req.chunk:
            ids = r.add_texts_chunked(req.text, req.metadata)
            doc_id = ids[0] if ids else ""
        else:
            doc_id = r.add_document(req.text, req.metadata)
        return IndexResponse(id=doc_id)

    @app.post("/index/batch", response_model=IndexBatchResponse)
    def index_batch(req: IndexBatchRequest):
        verify_key()
        r = get_rag()
        ids = r.add_documents(req.texts, req.metadatas)
        return IndexBatchResponse(ids=ids, count=len(ids))

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        verify_key()
        r = get_rag()
        hits = r.search(req.query, k=req.k, filter_ids=req.filter_ids)
        results = [
            SearchResult(id=h.id, text=h.text, score=h.score, metadata=h.metadata)
            for h in hits
        ]
        return SearchResponse(results=results, query=req.query, count=len(results))

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        verify_key()
        r = get_rag()
        try:
            answer, sources = r.ask(req.question, k=req.k, system=req.system)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        src = [
            SearchResult(id=s.id, text=s.text, score=s.score, metadata=s.metadata)
            for s in sources
        ]
        return AskResponse(answer=answer, sources=src, question=req.question)

    @app.delete("/document/{doc_id}")
    def delete_doc(doc_id: str):
        verify_key()
        r = get_rag()
        removed = r.delete(doc_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"id": doc_id, "status": "deleted"}

    return app


# ---------------------------------------------------------------------------
# Default app instance (for uvicorn)
# ---------------------------------------------------------------------------
app = create_app()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from ..config import TurboRagConfig
    cfg = TurboRagConfig()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "turborag.api.server:app",
        host=cfg.api.host,
        port=cfg.api.port,
        workers=cfg.api.workers,
        reload=False,
    )
