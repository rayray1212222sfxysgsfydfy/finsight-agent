"""Unit tests for src/tools/db_tools.py.

All tests use a shared in-memory SQLite database — no real file writes. The
keeper connection in the fixture holds the in-memory DB alive so that the
separate connections opened by each tool call observe the same data.
"""

import sqlite3

import pytest

from src.tools import db_tools


@pytest.fixture
def in_memory_db(monkeypatch):
    """Point db_tools at a shared in-memory SQLite database for one test."""
    shared_uri = "file::memory:?cache=shared"
    monkeypatch.setattr(db_tools, "DB_PATH", shared_uri)
    keeper = sqlite3.connect(shared_uri, uri=True)
    yield keeper
    keeper.close()


def test_write_and_read_back_value_succeeds(in_memory_db):
    write_result = db_tools.write_to_db("pnc_2023", "report-data")
    assert write_result["error"] is None

    read_result = db_tools.read_from_db("pnc_2023")
    assert read_result["error"] is None
    assert read_result["data"] == "report-data"


def test_check_cache_missing_key_returns_none(in_memory_db):
    result = db_tools.check_cache("does_not_exist")
    assert result["error"] is None
    assert result["data"] is None


def test_check_cache_existing_key_returns_value(in_memory_db):
    db_tools.write_to_db("cached_key", "cached_value")

    result = db_tools.check_cache("cached_key")
    assert result["error"] is None
    assert result["data"] == "cached_value"


def test_write_to_db_empty_key_raises_value_error(in_memory_db):
    with pytest.raises(ValueError):
        db_tools.write_to_db("", "value")


def test_read_from_db_empty_key_raises_value_error(in_memory_db):
    with pytest.raises(ValueError):
        db_tools.read_from_db("")


def test_check_cache_empty_key_raises_value_error(in_memory_db):
    with pytest.raises(ValueError):
        db_tools.check_cache("")
