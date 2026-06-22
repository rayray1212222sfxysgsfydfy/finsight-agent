"""Retrieval-augmented generation tools backed by ChromaDB.

Embeds filing text into the `finsight_filings` collection and queries it for
relevant chunks. Following the error-handling contract, these functions never
raise except for ValueError on invalid input (empty text/query) before any I/O.

ChromaDB caveat: PersistentClient silently fails if its path directory does not
exist, so _get_collection() creates data/chromadb/ before instantiating.
"""

from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import chromadb

CHROMA_PATH = str(Path("data") / "chromadb")
COLLECTION_NAME = "finsight_filings"


def _get_collection() -> Any:
    """Return the finsight_filings collection, creating the store if needed.

    Returns:
        a ChromaDB collection object for COLLECTION_NAME
    """
    # PersistentClient silently fails if the directory is missing — create it.
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def embed_text(text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Embed a text chunk into the vector store with attached metadata.

    Args:
        text: non-empty document text to embed
        metadata: metadata dict to associate with the chunk (e.g. ticker, year)

    Returns:
        {'data': {'doc_id': str}, 'error': None} on success
        {'data': None, 'error': 'message'} on failure
    """
    if not text or not isinstance(text, str):
        raise ValueError("text must be a non-empty string")

    try:
        collection = _get_collection()
        doc_id = uuid4().hex
        collection.add(
            documents=[text],
            metadatas=[metadata] if metadata else None,
            ids=[doc_id],
        )
        return {"data": {"doc_id": doc_id}, "error": None}
    except Exception as exc:
        return {"data": None, "error": f"Failed to embed text: {exc}"}


def query_vector_store(query: str, n_results: int = 5) -> Dict[str, Any]:
    """Query the vector store for chunks most relevant to the query.

    Args:
        query: non-empty natural-language query string
        n_results: maximum number of chunks to return

    Returns:
        {'data': [{'text', 'metadata', 'id', 'distance'}, ...], 'error': None}
        on success; {'data': None, 'error': 'message'} on failure
    """
    if not query or not isinstance(query, str):
        raise ValueError("query must be a non-empty string")

    try:
        collection = _get_collection()
        results = collection.query(query_texts=[query], n_results=n_results)

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        chunks: List[Dict[str, Any]] = []
        for i, document in enumerate(documents):
            chunks.append(
                {
                    "text": document,
                    "metadata": metadatas[i] if i < len(metadatas) else None,
                    "id": ids[i] if i < len(ids) else None,
                    "distance": distances[i] if i < len(distances) else None,
                }
            )
        return {"data": chunks, "error": None}
    except Exception as exc:
        return {"data": None, "error": f"Failed to query vector store: {exc}"}


__all__ = ["embed_text", "query_vector_store"]
