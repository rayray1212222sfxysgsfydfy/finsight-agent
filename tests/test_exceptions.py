"""Tests for src/utils/exceptions.py."""

import pytest

from src.utils.exceptions import AgentExecutionError


def test_agent_execution_error_is_exception_subclass():
    assert issubclass(AgentExecutionError, Exception)


def test_agent_execution_error_raises_and_carries_message():
    with pytest.raises(AgentExecutionError) as exc_info:
        raise AgentExecutionError("ingest failed unrecoverably")

    assert "ingest failed unrecoverably" in str(exc_info.value)
