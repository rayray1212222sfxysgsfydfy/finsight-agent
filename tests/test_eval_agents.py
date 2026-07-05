"""Pytest wrappers for Tier 2 agent contract evals (src/evals/eval_agents.py)."""

import pytest
from src.evals.eval_agents import _EVALS


@pytest.mark.parametrize("eval_fn", _EVALS, ids=lambda f: f.__name__)
def test_agent_eval(eval_fn):
    eval_fn()
