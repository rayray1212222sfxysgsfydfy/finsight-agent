"""Structured agent-trace logging.

Writes one JSON trace record per agent/tool invocation into the SQLite store
(via db_tools.write_to_db) so pipeline runs are auditable. No LLM calls and no
agent logic live here. Follows the CLAUDE.md error-handling contract: ValueError
only for invalid input before any I/O; otherwise returns {'data': ..., 'error': ...}.

Note: db_tools.write_to_db persists to the key/value `cache` table, so each
trace is stored under an "agent_trace:{run_id}:{timestamp}" key with the JSON
record as its value — the logical `agent_traces` table from CLAUDE.md's layout.
"""

import datetime
import json
from typing import Any, Dict, Optional

from src.tools.db_tools import write_to_db


def log_agent_trace(
    run_id: str,
    agent_name: str,
    tool_name: Optional[str],
    input_summary: str,
    output_summary: str,
    latency_ms: float,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Write a structured trace record for one agent/tool invocation.

    Args:
        run_id: unique identifier for the pipeline run (required).
        agent_name: name of the agent emitting the trace (required).
        tool_name: name of the tool invoked, or None for an agent-level trace.
        input_summary: short human-readable summary of the input.
        output_summary: short human-readable summary of the output.
        latency_ms: wall-clock latency of the invocation in milliseconds.
        timestamp: ISO-8601 timestamp; generated (UTC) when omitted.

    Returns:
        {'data': key, 'error': None} on success
        {'data': None, 'error': 'message'} on failure
    """
    if not run_id or not isinstance(run_id, str):
        raise ValueError("run_id must be a non-empty string")
    if not agent_name or not isinstance(agent_name, str):
        raise ValueError("agent_name must be a non-empty string")

    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    record = {
        "run_id": run_id,
        "agent_name": agent_name,
        "tool_name": tool_name,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "latency_ms": latency_ms,
        "timestamp": timestamp,
    }

    try:
        payload = json.dumps(record)
    except (TypeError, ValueError) as exc:
        return {"data": None, "error": f"Failed to serialize trace: {exc}"}

    key = f"agent_trace:{run_id}:{timestamp}"
    return write_to_db(key, payload)


__all__ = ["log_agent_trace"]
