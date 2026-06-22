"""SQLite-backed key/value storage and caching tools.

Provides a simple persistent key/value store over SQLite used for caching
fetched data and intermediate pipeline results. Following the error-handling
contract, these functions never raise except for ValueError on invalid input
(an empty key) before any I/O.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict

# Default on-disk location. Tests monkeypatch this to a shared in-memory URI.
DB_PATH = str(Path("data") / "finsight.db")


def _get_connection() -> sqlite3.Connection:
    """Open a connection to DB_PATH and ensure the cache table exists.

    Returns:
        an open sqlite3 connection with the `cache` table guaranteed present
    """
    uri = DB_PATH.startswith("file:")
    conn = sqlite3.connect(DB_PATH, uri=uri)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
    )
    return conn


def write_to_db(key: str, value: str) -> Dict[str, Any]:
    """Write (insert or replace) a key/value pair into the store.

    Args:
        key: non-empty string key
        value: string value to store

    Returns:
        {'data': key, 'error': None} on success
        {'data': None, 'error': 'message'} on failure
    """
    if not key or not isinstance(key, str):
        raise ValueError("key must be a non-empty string")

    try:
        conn = _get_connection()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()
        finally:
            conn.close()
        return {"data": key, "error": None}
    except Exception as exc:
        return {"data": None, "error": f"Failed to write key '{key}': {exc}"}


def read_from_db(key: str) -> Dict[str, Any]:
    """Read the value for a key from the store.

    Args:
        key: non-empty string key

    Returns:
        {'data': value, 'error': None} when the key exists
        {'data': None, 'error': 'message'} when missing or on failure
    """
    if not key or not isinstance(key, str):
        raise ValueError("key must be a non-empty string")

    try:
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return {"data": None, "error": f"Failed to read key '{key}': {exc}"}

    if row is None:
        return {"data": None, "error": f"Key '{key}' not found"}
    return {"data": row[0], "error": None}


def check_cache(key: str) -> Dict[str, Any]:
    """Look up a cached value, treating a miss as a non-error.

    Unlike read_from_db, a missing key is not an error — it returns
    {'data': None, 'error': None} so callers can branch on a cache miss.

    Args:
        key: non-empty string key

    Returns:
        {'data': value, 'error': None} on a cache hit
        {'data': None, 'error': None} on a cache miss
        {'data': None, 'error': 'message'} on failure
    """
    if not key or not isinstance(key, str):
        raise ValueError("key must be a non-empty string")

    try:
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return {"data": None, "error": f"Failed to check cache for '{key}': {exc}"}

    if row is None:
        return {"data": None, "error": None}
    return {"data": row[0], "error": None}


__all__ = ["write_to_db", "read_from_db", "check_cache"]
