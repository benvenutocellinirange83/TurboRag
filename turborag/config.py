"""
TurboRag Configuration
======================
Central settings for all TurboRag components.
All values can be overridden via environment variables or passed directly.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmbedderConfig:
    """Config for the local GGUF embedding model (Gemma 270M by default)."""
    model_path: str = os.getenv(
        "TURBORAG_EMBED_MODEL",
        "models/gemma-embedding-270m-Q4_K_M.gguf",
    )
    n_ctx: int = int(os.getenv("TURBORAG_EMBED_CTX", "512"))
    n_threads: int = int(os.getenv("TURBORAG_EMBED_THREADS", "4"))
    n_gpu_layers: int = int(os.getenv("TURBORAG_EMBED_GPU_LAYERS", "0"))
    # Gemma embedding 270M outputs 2048-dim vectors
    dim: int = int(os.getenv("TURBORAG_EMBED_DIM", "2048"))
    # Normalize embeddings to unit sphere for cosine similarity
    normalize: bool = True


@dataclass
class LLMConfig:
    """Config for the local GGUF language model."""
    model_path: str = os.getenv(
        "TURBORAG_LLM_MODEL",
        "models/qwen-0.5b-Q4_K_M.gguf",
    )
    n_ctx: int = int(os.getenv("TURBORAG_LLM_CTX", "2048"))
    n_threads: int = int(os.getenv("TURBORAG_LLM_THREADS", "4"))
    n_gpu_layers: int = int(os.getenv("TURBORAG_LLM_GPU_LAYERS", "0"))
    max_tokens: int = int(os.getenv("TURBORAG_LLM_MAX_TOKENS", "512"))
    temperature: float = float(os.getenv("TURBORAG_LLM_TEMP", "0.1"))
    # Chat template: "chatml" (Qwen/SmolVLM) | "deepseek" | "llama"
    chat_template: str = os.getenv("TURBORAG_LLM_TEMPLATE", "chatml")


@dataclass
class IndexConfig:
    """Config for TurboVec quantized index."""
    # dim must match embedder output
    dim: int = int(os.getenv("TURBORAG_INDEX_DIM", "2048"))
    # 4-bit quantization: best balance of speed and recall
    bit_width: int = int(os.getenv("TURBORAG_INDEX_BITS", "4"))
    index_path: str = os.getenv("TURBORAG_INDEX_PATH", "data/turborag.tvim")
    docstore_path: str = os.getenv("TURBORAG_DOCSTORE_PATH", "data/docstore.db")


@dataclass
class APIConfig:
    """Config for the TurboRag FastAPI server."""
    host: str = os.getenv("TURBORAG_API_HOST", "127.0.0.1")
    port: int = int(os.getenv("TURBORAG_API_PORT", "8000"))
    workers: int = int(os.getenv("TURBORAG_API_WORKERS", "1"))
    api_key: Optional[str] = os.getenv("TURBORAG_API_KEY", None)
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class MCPConfig:
    """Config for the TurboRag MCP server."""
    host: str = os.getenv("TURBORAG_MCP_HOST", "127.0.0.1")
    port: int = int(os.getenv("TURBORAG_MCP_PORT", "8001"))
    name: str = "TurboRag MCP Server"


@dataclass
class TurboRagConfig:
    """Master config — compose sub-configs here."""
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    api: APIConfig = field(default_factory=APIConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    # RAG retrieval: number of top-k documents to retrieve
    top_k: int = int(os.getenv("TURBORAG_TOP_K", "5"))
    # Chunk settings for document ingestion
    chunk_size: int = int(os.getenv("TURBORAG_CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("TURBORAG_CHUNK_OVERLAP", "80"))

    @classmethod
    def from_env(cls) -> "TurboRagConfig":
        """Create config purely from environment variables."""
        return cls()


# Default singleton
DEFAULT_CONFIG = TurboRagConfig()

# Known GGUF model presets
MODELS = {
    # Embedding models
    "gemma-embedding-270m":    {"type": "embed",  "dim": 2048, "ctx": 512},
    # LLM models (generation)
    "qwen-0.5b":               {"type": "llm",    "template": "chatml", "ctx": 2048},
    "deepseek-1.3b":           {"type": "llm",    "template": "deepseek", "ctx": 4096},
    "smolvlm-135m":            {"type": "vlm",    "template": "chatml", "ctx": 2048},
    "smolvlm-256m":            {"type": "vlm",    "template": "chatml", "ctx": 2048},
    "smolvlm-500m":            {"type": "vlm",    "template": "chatml", "ctx": 4096},
}
