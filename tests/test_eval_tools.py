"""Pytest wrappers for Tier 1 tool evals (src/evals/eval_tools.py)."""

import pytest
from src.evals.eval_tools import _EVALS


@pytest.mark.parametrize("eval_fn", _EVALS, ids=lambda f: f.__name__)
def test_tool_eval(eval_fn):
    eval_fn()
