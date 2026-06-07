"""
TurboRag Core
=============
Main TurboRag class: document ingestion, quantized vector index, retrieval,
and end-to-end RAG generation — all offline, low CPU, low RAM.

Architecture
------------
  Text → Embedder (Gemma 270M GGUF)
       → IdMapIndex (TurboVec Q4, ~8× compression vs float32)
       → search(query, k=5)
       → LLM (Qwen/DeepSeek/SmolVLM GGUF)
       → answer

Persistence
-----------
  - Vector index: data/turborag.tvim   (TurboVec binary format)
  - Docstore:     data/docstore.db     (SQLite — metadata + text)
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import TurboRagConfig
from .embedder import Embedder
from .llm import LLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A chunk of text with its metadata."""
    id: str
    text: str
    metadata: Dict[str, Any]

    @classmethod
    def from_text(cls, text: str, metadata: Optional[Dict[str, Any]] = None) -> "Document":
        doc_id = hashlib.md5(text.encode()).hexdigest()
        return cls(id=doc_id, text=text, metadata=metadata or {})


@dataclass
class SearchResult:
    """One hit returned by TurboRag.search()."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# TurboRag
# ---------------------------------------------------------------------------

class TurboRag:
    """Quantized local RAG engine backed by TurboVec + SQLite.

    Parameters
    ----------
    config : TurboRagConfig
        Full configuration object.  Can be left default for a quick start.
    embedder : Embedder, optional
        Supply your own embedder to avoid loading the default Gemma model.
    llm : LLM, optional
        Supply your own LLM.  If None, generation methods raise RuntimeError.
    """

    def __init__(
        self,
        config: Optional[TurboRagConfig] = None,
        embedder: Optional[Embedder] = None,
        llm: Optional[LLM] = None,
    ) -> None:
        self.config = config or TurboRagConfig()
        self.embedder = embedder or Embedder.from_config(self.config.embedder)
        self.llm = llm

        # Ensure data directories exist
        Path(self.config.index.index_path).parent.mkdir(parents=True, exist_ok=True)

        # Vector index (lazy turbovec import so the package works without it)
        self._index = None
        self._id_counter: int = 0
        self._str_to_u64: Dict[str, int] = {}
        self._u64_to_str: Dict[int, str] = {}

        # SQLite docstore
        self._db = self._open_db(self.config.index.docstore_path)
        self._restore_id_map()

        # Try loading an existing index from disk
        self._load_index_if_exists()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        embed_model: str = "models/gemma-embedding-270m-Q4_K_M.gguf",
        llm_model: str = "models/qwen-0.5b-Q4_K_M.gguf",
        index_path: str = "data/turborag.tvim",
        dim: int = 2048,
        bit_width: int = 4,
    ) -> "TurboRag":
        """Convenience constructor — minimal boilerplate."""
        from .config import EmbedderConfig, LLMConfig, IndexConfig, TurboRagConfig
        cfg = TurboRagConfig(
            embedder=EmbedderConfig(model_path=embed_model, dim=dim),
            llm=LLMConfig(model_path=llm_model),
            index=IndexConfig(dim=dim, bit_width=bit_width, index_path=index_path),
        )
        llm = LLM.from_config(cfg.llm)
        return cls(config=cfg, llm=llm)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a single document.  Returns its string ID."""
        doc = Document.from_text(text, metadata)
        self._add_docs([doc])
        return doc.id

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """Add a batch of texts.  Returns list of IDs."""
        metas = metadatas or [{} for _ in texts]
        docs = [Document.from_text(t, m) for t, m in zip(texts, metas)]
        self._add_docs(docs)
        return [d.id for d in docs]

    def add_texts_chunked(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[str]:
        """Split a long text into overlapping chunks, then index them."""
        size = chunk_size or self.config.chunk_size
        overlap = chunk_overlap or self.config.chunk_overlap
        chunks = self._chunk_text(text, size, overlap)
        meta = metadata or {}
        ids = []
        for i, chunk in enumerate(chunks):
            chunk_meta = {**meta, "chunk_index": i, "total_chunks": len(chunks)}
            ids.append(self.add_document(chunk, chunk_meta))
        return ids

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Semantic search.  Returns up to k results, highest score first."""
        index = self._get_index()
        if len(index) == 0:
            return []

        top_k = k or self.config.top_k
        qvec = self.embedder.embed(query).reshape(1, -1)

        kwargs: Dict[str, Any] = {}
        if filter_ids:
            allowlist = np.array(
                [self._str_to_u64[fid] for fid in filter_ids if fid in self._str_to_u64],
                dtype=np.uint64,
            )
            if len(allowlist) > 0:
                kwargs["allowlist"] = allowlist

        scores_arr, ids_arr = index.search(qvec, top_k, **kwargs)
        scores = scores_arr[0].tolist()
        u64_ids = ids_arr[0].tolist()

        results = []
        for score, u64 in zip(scores, u64_ids):
            str_id = self._u64_to_str.get(int(u64))
            if str_id is None:
                continue
            row = self._db_get(str_id)
            if row is None:
                continue
            text, meta_json = row
            import json
            results.append(SearchResult(
                id=str_id,
                text=text,
                score=float(score),
                metadata=json.loads(meta_json),
            ))
        return results

    def search_raw(self, vector: np.ndarray, k: int = 5) -> List[SearchResult]:
        """Search by raw embedding vector (bypasses the text→embed step)."""
        index = self._get_index()
        if len(index) == 0:
            return []
        q = vector.reshape(1, -1).astype(np.float32)
        scores_arr, ids_arr = index.search(q, k)
        scores = scores_arr[0].tolist()
        u64_ids = ids_arr[0].tolist()
        results = []
        for score, u64 in zip(scores, u64_ids):
            str_id = self._u64_to_str.get(int(u64))
            if str_id is None:
                continue
            row = self._db_get(str_id)
            if row:
                import json
                text, meta_json = row
                results.append(SearchResult(
                    id=str_id, text=text,
                    score=float(score),
                    metadata=json.loads(meta_json),
                ))
        return results

    # ------------------------------------------------------------------
    # RAG generation
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        k: Optional[int] = None,
        system: str = "You are a helpful, accurate assistant. Answer based on the provided context only.",
    ) -> Tuple[str, List[SearchResult]]:
        """Full RAG pipeline: retrieve → generate.  Returns (answer, sources)."""
        if self.llm is None:
            raise RuntimeError("No LLM configured.  Pass llm= to TurboRag.create().")
        results = self.search(question, k=k)
        context = [r.text for r in results]
        answer = self.llm.rag_answer(question, context, system=system)
        return answer, results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the vector index to disk."""
        idx = self._get_index()
        idx.write(self.config.index.index_path)
        logger.info("Index saved to %s  (n=%d)", self.config.index.index_path, len(idx))

    def load(self) -> None:
        """Reload index from disk."""
        self._load_index_if_exists()

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete(self, doc_id: str) -> bool:
        """Remove a document by ID from both index and docstore."""
        u64 = self._str_to_u64.get(doc_id)
        if u64 is None:
            return False
        index = self._get_index()
        removed = index.remove(u64)
        if removed:
            del self._str_to_u64[doc_id]
            del self._u64_to_str[u64]
            self._db_delete(doc_id)
        return removed

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def doc_count(self) -> int:
        index = self._get_index()
        return len(index)

    def stats(self) -> Dict[str, Any]:
        index = self._get_index()
        return {
            "doc_count": len(index),
            "dim": index.dim,
            "bit_width": index.bit_width,
            "index_path": self.config.index.index_path,
            "embed_model": self.config.embedder.model_path,
            "llm_model": self.config.llm.model_path if self.llm else None,
        }

    # ------------------------------------------------------------------
    # Internal: TurboVec index
    # ------------------------------------------------------------------

    def _get_index(self):
        if self._index is None:
            try:
                from turbovec import IdMapIndex
            except ImportError:
                raise ImportError(
                    "turbovec is required.  "
                    "Build from source: cd turbovec-main && pip install maturin && "
                    "cd turbovec-python && maturin develop --release"
                )
            self._index = IdMapIndex(bit_width=self.config.index.bit_width)
        return self._index

    def _load_index_if_exists(self) -> None:
        path = self.config.index.index_path
        if not os.path.exists(path):
            return
        try:
            from turbovec import IdMapIndex
            self._index = IdMapIndex.load(path)
            logger.info("Loaded index from %s  (n=%d)", path, len(self._index))
        except Exception as exc:
            logger.warning("Could not load index from %s: %s", path, exc)

    def _add_docs(self, docs: List[Document]) -> None:
        import json
        if not docs:
            return
        texts = [d.text for d in docs]
        vecs = self.embedder.embed_batch(texts)   # (n, dim) float32
        index = self._get_index()

        new_ids: List[int] = []
        for doc in docs:
            if doc.id in self._str_to_u64:
                # Already exists — skip (idempotent)
                new_ids.append(self._str_to_u64[doc.id])
                continue
            u64 = self._id_counter
            self._id_counter += 1
            self._str_to_u64[doc.id] = u64
            self._u64_to_str[u64] = doc.id
            new_ids.append(u64)

        u64_arr = np.array(new_ids, dtype=np.uint64)
        index.add_with_ids(vecs, u64_arr)

        for doc in docs:
            self._db_upsert(doc.id, doc.text, json.dumps(doc.metadata))

        logger.debug("Added %d documents to index (total=%d)", len(docs), len(index))

    # ------------------------------------------------------------------
    # Internal: SQLite docstore
    # ------------------------------------------------------------------

    def _open_db(self, path: str) -> sqlite3.Connection:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path, check_same_thread=False)
        con.execute(
            "CREATE TABLE IF NOT EXISTS docs "
            "(id TEXT PRIMARY KEY, text TEXT, meta TEXT, ts REAL)"
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_docs_id ON docs(id)")
        con.commit()
        return con

    def _db_upsert(self, doc_id: str, text: str, meta: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO docs (id, text, meta, ts) VALUES (?,?,?,?)",
            (doc_id, text, meta, time.time()),
        )
        self._db.commit()

    def _db_get(self, doc_id: str) -> Optional[Tuple[str, str]]:
        cur = self._db.execute("SELECT text, meta FROM docs WHERE id=?", (doc_id,))
        row = cur.fetchone()
        return row  # (text, meta_json) or None

    def _db_delete(self, doc_id: str) -> None:
        self._db.execute("DELETE FROM docs WHERE id=?", (doc_id,))
        self._db.commit()

    def _restore_id_map(self) -> None:
        """Rebuild _str_to_u64 from SQLite on startup (order = rowid)."""
        cur = self._db.execute("SELECT id FROM docs ORDER BY rowid")
        for i, (doc_id,) in enumerate(cur.fetchall()):
            self._str_to_u64[doc_id] = i
            self._u64_to_str[i] = doc_id
        self._id_counter = len(self._str_to_u64)

    # ------------------------------------------------------------------
    # Internal: text chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += size - overlap
        return chunks
