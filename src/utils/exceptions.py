"""Custom exceptions for FinSight.

Agent classes (src/agents/) raise AgentExecutionError for unrecoverable
pipeline failures. Tool functions (src/tools/) never raise it — they return
{'data': ..., 'error': ...} per the CLAUDE.md error-handling contract.
"""


class AgentExecutionError(Exception):
    """Raised by agent classes when a pipeline stage fails unrecoverably.

    Use this for agent-level failures the agent cannot recover from (e.g. a
    required tool returned an error and there is no fallback). Tool functions
    in src/tools/ must never raise this; they return an error dict instead,
    per the CLAUDE.md error-handling contract.
    """
