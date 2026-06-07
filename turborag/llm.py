"""
TurboRag LLM Wrapper
====================
Unified interface to all supported local GGUF language models:

  ┌─────────────────────┬──────────┬──────────┬────────────────┐
  │ Model               │ Params   │ Template │ Best For       │
  ├─────────────────────┼──────────┼──────────┼────────────────┤
  │ Qwen 0.5B Q4_K_M   │ ~300MB   │ chatml   │ Fast Q&A       │
  │ DeepSeek 1.3B Q4_KM │ ~750MB   │ deepseek │ Reasoning      │
  │ SmolVLM 135M Q4_KM │ ~90MB    │ chatml   │ Ultra-light    │
  │ SmolVLM 256M Q4_KM │ ~160MB   │ chatml   │ Balanced VLM   │
  │ SmolVLM 500M Q4_KM │ ~320MB   │ chatml   │ Vision+Text    │
  └─────────────────────┴──────────┴──────────┴────────────────┘

All run offline via llama-cpp-python (CPU, low RAM).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional

from .config import LLMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------

def _format_chatml(system: str, user: str) -> str:
    """ChatML format — used by Qwen & SmolVLM."""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _format_deepseek(system: str, user: str) -> str:
    """DeepSeek chat format."""
    if system:
        return f"<|begin_of_sentence|>{system}\n\nUser: {user}\n\nAssistant:"
    return f"<|begin_of_sentence|>User: {user}\n\nAssistant:"


def _format_llama(system: str, user: str) -> str:
    """Llama-2 chat format (fallback)."""
    if system:
        return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
    return f"[INST] {user} [/INST]"


TEMPLATES = {
    "chatml":   _format_chatml,
    "deepseek": _format_deepseek,
    "llama":    _format_llama,
}


# ---------------------------------------------------------------------------
# LLM class
# ---------------------------------------------------------------------------

class LLM:
    """Local GGUF language model via llama-cpp-python.

    Parameters
    ----------
    config : LLMConfig
        Path to the .gguf file plus inference settings.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._llm = None  # lazy load
        self._fmt = TEMPLATES.get(config.chat_template, _format_chatml)
        logger.info("LLM configured: %s (template=%s)", config.model_path, config.chat_template)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: LLMConfig) -> "LLM":
        return cls(config)

    @classmethod
    def qwen_05b(cls, model_path: str = "models/qwen-0.5b-Q4_K_M.gguf") -> "LLM":
        return cls(LLMConfig(model_path=model_path, chat_template="chatml", n_ctx=2048))

    @classmethod
    def deepseek_13b(cls, model_path: str = "models/deepseek-1.3b-Q4_K_M.gguf") -> "LLM":
        return cls(LLMConfig(model_path=model_path, chat_template="deepseek", n_ctx=4096))

    @classmethod
    def smolvlm(cls, size: str = "256m", model_path: Optional[str] = None) -> "LLM":
        path = model_path or f"models/smolvlm-{size}-Q4_K_M.gguf"
        ctx = 4096 if size == "500m" else 2048
        return cls(LLMConfig(model_path=path, chat_template="chatml", n_ctx=ctx))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate a completion string."""
        formatted = self._fmt(system, prompt)
        return self._run(
            formatted,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            stop=stop or ["<|im_end|>", "<|end|>", "</s>"],
        )

    def stream(
        self,
        prompt: str,
        system: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """Stream tokens as they are generated."""
        formatted = self._fmt(system, prompt)
        llm = self._get_llm()
        cfg = self.config
        for chunk in llm(
            formatted,
            max_tokens=max_tokens or cfg.max_tokens,
            temperature=temperature if temperature is not None else cfg.temperature,
            stop=["<|im_end|>", "<|end|>", "</s>"],
            stream=True,
        ):
            text = chunk["choices"][0].get("text", "")
            if text:
                yield text

    def rag_answer(
        self,
        question: str,
        context_chunks: List[str],
        system: str = "You are a helpful assistant. Answer using only the provided context.",
    ) -> str:
        """Standard RAG: insert retrieved context then answer the question."""
        context = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
        )
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        return self.generate(prompt, system=system)

    def warmup(self) -> None:
        self._get_llm()
        logger.info("LLM warmed up.")

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
        cfg = self.config
        logger.info("Loading LLM from %s …", cfg.model_path)
        llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=False,
        )
        logger.info("LLM loaded (ctx=%d).", cfg.n_ctx)
        return llm

    def _run(self, prompt: str, max_tokens: int, temperature: float, stop: List[str]) -> str:
        llm = self._get_llm()
        out = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
        return out["choices"][0]["text"].strip()
