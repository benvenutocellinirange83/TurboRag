"""
TurboRag Embedder
=================
Wraps llama-cpp-python to produce float32 embeddings from any GGUF model.

Supported models (all Q4_K_M GGUF):
  - Gemma Embedding 270M  (dim=2048, best quality / size trade-off)

Usage::

    embedder = Embedder.from_config(config.embedder)
    vec = embedder.embed("Hello, world!")          # np.ndarray (2048,)
    vecs = embedder.embed_batch(["a", "b", "c"])   # np.ndarray (3, 2048)
"""

from __future__ import annotations

import logging
from typing import List, Union
import numpy as np

from .config import EmbedderConfig

logger = logging.getLogger(__name__)


class Embedder:
    """GGUF-powered text embedder via llama-cpp-python.

    Keeps the model loaded in RAM between calls so repeated queries
    are fast.  Uses CPU-only by default (n_gpu_layers=0) to stay
    within low-RAM / low-CPU constraints.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config
        self._llm = None   # lazy load
        logger.info("Embedder configured: %s (dim=%d)", config.model_path, config.dim)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: EmbedderConfig) -> "Embedder":
        return cls(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string → (dim,) float32 ndarray."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings → (n, dim) float32 ndarray.

        Iterates one-by-one because llama-cpp batch embedding can OOM
        on low-RAM devices with long texts.
        """
        llm = self._get_llm()
        results = []
        for text in texts:
            raw = llm.embed(text)
            vec = np.array(raw, dtype=np.float32)
            if self.config.normalize:
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
            results.append(vec)
        return np.stack(results, axis=0)

    @property
    def dim(self) -> int:
        return self.config.dim

    def warmup(self) -> None:
        """Force-load the model now (instead of on first embed call)."""
        self._get_llm()
        logger.info("Embedder warmed up.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_llm(self):
        if self._llm is None:
            self._llm = self._load()
        return self._llm

    def _load(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required: pip install llama-cpp-python"
            )
        logger.info("Loading embedding model from %s …", self.config.model_path)
        llm = Llama(
            model_path=self.config.model_path,
            embedding=True,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=False,
        )
        logger.info("Embedding model loaded.")
        return llm


# ---------------------------------------------------------------------------
# Stub embedder — used in tests / when no GGUF model is available
# ---------------------------------------------------------------------------

class StubEmbedder(Embedder):
    """Deterministic random embedder for unit tests and CI."""

    def __init__(self, dim: int = 2048) -> None:
        from .config import EmbedderConfig
        cfg = EmbedderConfig(model_path="__stub__", dim=dim, normalize=True)
        # Skip Embedder.__init__ which would try to log
        self.config = cfg
        self._llm = None
        self._rng = np.random.default_rng(42)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        vecs = self._rng.standard_normal((len(texts), self.config.dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)
