"""Unit tests for src/tools/rag_tools.py.

All ChromaDB access is mocked via the module-level _get_collection() helper —
no real vector store is created or queried in tests.
"""

from unittest.mock import MagicMock

import pytest

from src.tools import rag_tools


def test_embed_text_success_returns_doc_id(mocker):
    fake_collection = MagicMock()
    mocker.patch(
        "src.tools.rag_tools._get_collection", return_value=fake_collection
    )

    result = rag_tools.embed_text("some 10-K filing text", {"ticker": "PNC", "year": 2023})

    assert result["error"] is None
    assert isinstance(result["data"], dict)
    assert "doc_id" in result["data"]
    assert isinstance(result["data"]["doc_id"], str)
    assert result["data"]["doc_id"]
    fake_collection.add.assert_called_once()


def test_embed_text_empty_text_raises_value_error():
    with pytest.raises(ValueError):
        rag_tools.embed_text("", {"ticker": "PNC"})


def test_query_vector_store_returns_list_of_chunks(mocker):
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [["chunk one", "chunk two"]],
        "metadatas": [[{"ticker": "PNC"}, {"ticker": "PNC"}]],
        "ids": [["id1", "id2"]],
        "distances": [[0.1, 0.2]],
    }
    mocker.patch(
        "src.tools.rag_tools._get_collection", return_value=fake_collection
    )

    result = rag_tools.query_vector_store("liquidity risk", n_results=2)

    assert result["error"] is None
    assert isinstance(result["data"], list)
    assert len(result["data"]) == 2


def test_query_vector_store_empty_query_raises_value_error():
    with pytest.raises(ValueError):
        rag_tools.query_vector_store("")
