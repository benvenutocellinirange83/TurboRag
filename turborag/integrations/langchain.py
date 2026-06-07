"""
TurboRag × LangChain Integration
==================================
Drop-in LangChain VectorStore backed by TurboRag.
Uses the turbovec IdMapIndex for ultra-compressed storage.

Usage::

    from turborag.integrations.langchain import TurboRagVectorStore
    from langchain_community.embeddings import LlamaCppEmbeddings

    embeddings = LlamaCppEmbeddings(model_path="models/gemma-embedding-270m-Q4_K_M.gguf")
    store = TurboRagVectorStore(embeddings=embeddings)
    store.add_texts(["doc1", "doc2"])
    docs = store.similarity_search("my query", k=3)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
    _LANGCHAIN = True
except ImportError:
    _LANGCHAIN = False
    Document = None
    Embeddings = None
    VectorStore = object


class TurboRagVectorStore(VectorStore):  # type: ignore[misc]
    """LangChain VectorStore backed by TurboRag (TurboVec + SQLite).

    This class is a thin adapter that lets you use TurboRag inside any
    LangChain pipeline or agent that expects a ``VectorStore``.

    Parameters
    ----------
    embeddings : langchain Embeddings
        Any LangChain Embeddings object (e.g. LlamaCppEmbeddings,
        HuggingFaceEmbeddings, etc.).
    rag : TurboRag, optional
        Pre-built TurboRag instance.  If None, one is created with defaults.
    """

    def __init__(
        self,
        embeddings: "Embeddings",
        rag=None,
        **kwargs: Any,
    ) -> None:
        if not _LANGCHAIN:
            raise ImportError(
                "langchain-core is required: pip install langchain-core"
            )
        self._embeddings = embeddings
        if rag is None:
            from ..core import TurboRag
            from ..embedder import Embedder

            class _LCEmbedWrapper:
                """Adapter: wraps a LangChain Embeddings as a TurboRag Embedder."""
                def __init__(self, lc_embed):
                    self._e = lc_embed

                def embed(self, text: str) -> np.ndarray:
                    return np.array(self._e.embed_query(text), dtype=np.float32)

                def embed_batch(self, texts: List[str]) -> np.ndarray:
                    vecs = self._e.embed_documents(texts)
                    return np.array(vecs, dtype=np.float32)

                @property
                def dim(self):
                    return None  # inferred from first add

            rag = TurboRag(embedder=_LCEmbedWrapper(embeddings))

        self._rag = rag

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        text_list = list(texts)
        return self._rag.add_documents(text_list, metadatas)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List["Document"]:
        results = self._rag.search(query, k=k)
        return [
            Document(page_content=r.text, metadata={**r.metadata, "_score": r.score, "_id": r.id})
            for r in results
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple["Document", float]]:
        results = self._rag.search(query, k=k)
        return [
            (
                Document(page_content=r.text, metadata={**r.metadata, "_id": r.id}),
                r.score,
            )
            for r in results
        ]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List["Document"]:
        vec = np.array(embedding, dtype=np.float32)
        results = self._rag.search_raw(vec, k=k)
        return [
            Document(page_content=r.text, metadata={**r.metadata, "_id": r.id})
            for r in results
        ]

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            return False
        for doc_id in ids:
            self._rag.delete(doc_id)
        return True

    def save(self, path: str) -> None:
        self._rag.config.index.index_path = path
        self._rag.save()

    @classmethod
    def load(cls, path: str, embeddings: "Embeddings") -> "TurboRagVectorStore":
        from ..core import TurboRag
        from ..config import TurboRagConfig, IndexConfig
        cfg = TurboRagConfig(index=IndexConfig(index_path=path))
        rag = TurboRag(config=cfg)
        return cls(embeddings=embeddings, rag=rag)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: "Embeddings",
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "TurboRagVectorStore":
        store = cls(embeddings=embedding, **kwargs)
        store.add_texts(texts, metadatas)
        return store

    @property
    def embeddings(self) -> "Embeddings":
        return self._embeddings


# ---------------------------------------------------------------------------
# LangChain LlamaCpp embeddings helper
# ---------------------------------------------------------------------------

def make_llamacpp_embeddings(model_path: str, n_ctx: int = 512, n_threads: int = 4):
    """Create a LangChain Embeddings object backed by a GGUF model.

    Example::

        emb = make_llamacpp_embeddings("models/gemma-embedding-270m-Q4_K_M.gguf")
        store = TurboRagVectorStore(embeddings=emb)
    """
    try:
        from langchain_community.embeddings import LlamaCppEmbeddings
        return LlamaCppEmbeddings(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
        )
    except ImportError:
        raise ImportError(
            "langchain-community is required: pip install langchain-community"
        )


# ---------------------------------------------------------------------------
# LangChain LlamaCpp LLM helper
# ---------------------------------------------------------------------------

def make_llamacpp_llm(
    model_path: str,
    n_ctx: int = 2048,
    n_threads: int = 4,
    max_tokens: int = 512,
    temperature: float = 0.1,
    chat_format: str = "chatml",
):
    """Create a LangChain LLM backed by a GGUF model via llama.cpp.

    Example::

        llm = make_llamacpp_llm("models/qwen-0.5b-Q4_K_M.gguf")
        # use in a ConversationalRetrievalChain, Agent, etc.
    """
    try:
        from langchain_community.llms import LlamaCpp
        return LlamaCpp(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except ImportError:
        raise ImportError(
            "langchain-community is required: pip install langchain-community"
        )
