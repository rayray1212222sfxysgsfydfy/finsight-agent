"""Tests for src/utils/logger.py.

write_to_db is mocked where it is used (src.utils.logger.write_to_db) so no real
SQLite I/O happens during the test.
"""

import pytest

from src.utils.logger import log_agent_trace


def test_log_agent_trace_success_returns_no_error(mocker):
    mocker.patch(
        "src.utils.logger.write_to_db",
        return_value={"data": "agent_trace:run-1", "error": None},
    )

    result = log_agent_trace(
        run_id="run-1",
        agent_name="IngestAgent",
        tool_name="fetch_10k_filing",
        input_summary="PNC 2023",
        output_summary="revenue=22B",
        latency_ms=123.4,
    )

    assert result["error"] is None


def test_log_agent_trace_missing_run_id_raises_value_error():
    with pytest.raises(ValueError):
        log_agent_trace(
            run_id="",
            agent_name="IngestAgent",
            tool_name=None,
            input_summary="",
            output_summary="",
            latency_ms=0.0,
        )
