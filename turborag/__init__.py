"""
TurboRag
========
Local, quantized RAG engine — low CPU, low RAM, fully offline.

Quick start::

    from turborag import TurboRag

    rag = TurboRag.create(
        embed_model="models/gemma-embedding-270m-Q4_K_M.gguf",
        llm_model="models/qwen-0.5b-Q4_K_M.gguf",
    )
    rag.add_document("Paris is the capital of France.")
    answer, sources = rag.ask("What is the capital of France?")
    print(answer)
"""

from .core import TurboRag, Document, SearchResult
from .embedder import Embedder, StubEmbedder
from .llm import LLM
from .config import TurboRagConfig

__all__ = [
    "TurboRag",
    "Document",
    "SearchResult",
    "Embedder",
    "StubEmbedder",
    "LLM",
    "TurboRagConfig",
]
__version__ = "1.0.0"
