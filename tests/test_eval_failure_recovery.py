"""Pytest wrappers for failure/recovery evals (src/evals/eval_failure_recovery.py)."""

import pytest
from src.evals.eval_failure_recovery import _EVALS


@pytest.mark.parametrize("eval_fn", _EVALS, ids=lambda f: f.__name__)
def test_failure_recovery_eval(eval_fn):
    eval_fn()
