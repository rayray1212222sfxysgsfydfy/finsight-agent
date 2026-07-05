"""Pytest wrappers for Tier 3 LLM-as-judge evals (src/evals/eval_llm_judge.py).

Uses mock mode (no real API calls) so these run in CI without credentials.
The live judge can be invoked via: python src/evals/eval_llm_judge.py
"""

import pytest
from src.evals.eval_llm_judge import NARRATIVE_SAMPLES, run_llm_judge_evals


def test_llm_judge_all_samples_scored():
    summary = run_llm_judge_evals(use_mock=True)
    assert summary["scored"] == summary["total"] == len(NARRATIVE_SAMPLES)


def test_llm_judge_average_score_meets_threshold():
    summary = run_llm_judge_evals(use_mock=True)
    assert summary["avg_overall"] >= 3.5, (
        f"Average score {summary['avg_overall']:.2f} below 3.5 threshold"
    )


def test_llm_judge_no_hallucination_violations():
    summary = run_llm_judge_evals(use_mock=True)
    assert summary["hallucination_violations"] == [], (
        f"Hallucination detected in: {summary['hallucination_violations']}"
    )


def test_llm_judge_threshold_met():
    summary = run_llm_judge_evals(use_mock=True)
    assert summary["threshold_met"], (
        f"Judge threshold not met: avg={summary['avg_overall']:.2f}, "
        f"violations={summary['hallucination_violations']}"
    )


@pytest.mark.parametrize("sample", NARRATIVE_SAMPLES, ids=lambda s: s["id"])
def test_llm_judge_sample_scores_valid(sample):
    """Each sample's mock score must meet its own minimum expectations."""
    summary = run_llm_judge_evals(use_mock=True)
    result = next(r for r in summary["samples"] if r["id"] == sample["id"])
    assert result.get("accuracy") is not None, f"{sample['id']} has no accuracy score"
    assert result.get("hallucination") == 5, (
        f"{sample['id']} hallucination score {result.get('hallucination')} != 5"
    )
