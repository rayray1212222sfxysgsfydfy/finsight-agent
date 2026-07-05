"""Pytest wrappers for Tier 2 pipeline integration evals (src/evals/eval_pipeline.py)."""

import pytest
from src.evals.eval_pipeline import _EVALS, PASS_THRESHOLD


@pytest.mark.parametrize("eval_fn", _EVALS, ids=lambda f: f.__name__)
def test_pipeline_eval(eval_fn):
    eval_fn()
