"""
TurboRag Python SDK
====================
Typed HTTP client for the TurboRag REST API.

Usage::

    from turborag_sdk import TurboRagClient

    client = TurboRagClient("http://127.0.0.1:8000")

    # Index documents
    doc_id = client.index("Paris is the capital of France.")

    # Search
    results = client.search("capital of France", k=3)
    for r in results:
        print(r.score, r.text)

    # Ask (RAG)
    resp = client.ask("What is the capital of France?")
    print(resp.answer)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import urllib.request
import urllib.error
import json


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class AskResponse:
    answer: str
    sources: List[SearchResult]
    question: str


@dataclass
class IndexResponse:
    id: str
    status: str


class TurboRagError(Exception):
    """Raised on API errors."""


class TurboRagClient:
    """HTTP client for the TurboRag REST API.

    Parameters
    ----------
    base_url : str
        Base URL of the running TurboRag API, e.g. ``http://127.0.0.1:8000``.
    api_key : str, optional
        Value for the ``X-API-Key`` header (if the server requires it).
    timeout : int
        Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self) -> bool:
        """Return True if the server is alive."""
        try:
            resp = self._get("/health")
            return resp.get("status") == "ok"
        except Exception:
            return False

    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        return self._get("/stats")

    def embed(self, text: str) -> List[float]:
        """Return the embedding vector for *text*."""
        resp = self._post("/embed", {"text": text})
        return resp["embedding"]

    def index(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = False,
    ) -> str:
        """Index a document.  Returns its ID."""
        resp = self._post("/index", {
            "text": text,
            "metadata": metadata or {},
            "chunk": chunk,
        })
        return resp["id"]

    def index_batch(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Index multiple documents.  Returns list of IDs."""
        resp = self._post("/index/batch", {
            "texts": texts,
            "metadatas": metadatas,
        })
        return resp["ids"]

    def search(
        self,
        query: str,
        k: int = 5,
        filter_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Semantic search.  Returns a list of SearchResult."""
        resp = self._post("/search", {
            "query": query,
            "k": k,
            "filter_ids": filter_ids,
        })
        return [
            SearchResult(
                id=r["id"],
                text=r["text"],
                score=r["score"],
                metadata=r["metadata"],
            )
            for r in resp["results"]
        ]

    def ask(self, question: str, k: int = 5, system: str = "") -> AskResponse:
        """Full RAG: retrieve + generate.  Returns AskResponse."""
        body: Dict[str, Any] = {"question": question, "k": k}
        if system:
            body["system"] = system
        resp = self._post("/ask", body)
        sources = [
            SearchResult(
                id=s["id"],
                text=s["text"],
                score=s["score"],
                metadata=s["metadata"],
            )
            for s in resp.get("sources", [])
        ]
        return AskResponse(
            answer=resp["answer"],
            sources=sources,
            question=resp["question"],
        )

    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID.  Returns True if removed."""
        try:
            self._delete(f"/document/{doc_id}")
            return True
        except TurboRagError:
            return False

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _get(self, path: str) -> Any:
        url = self.base_url + path
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        return self._execute(req)

    def _post(self, path: str, body: Any) -> Any:
        url = self.base_url + path
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        return self._execute(req)

    def _delete(self, path: str) -> Any:
        url = self.base_url + path
        req = urllib.request.Request(url, headers=self._headers(), method="DELETE")
        return self._execute(req)

    def _execute(self, req: urllib.request.Request) -> Any:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise TurboRagError(f"HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise TurboRagError(f"Connection error: {exc.reason}") from exc
