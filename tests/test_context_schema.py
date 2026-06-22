import pytest

from src.agents import context_schema


def test_base_context_has_exact_fields():
    assert list(context_schema.BaseContext.__annotations__.keys()) == [
        "ticker",
        "year",
        "run_id",
    ]


def test_ingest_output_has_all_expected_fields():
    expected_fields = {
        "revenue",
        "net_income",
        "total_assets",
        "total_liabilities",
        "shareholders_equity",
        "current_assets",
        "current_liabilities",
        "ebit",
        "interest_expense",
        "retained_earnings",
        "working_capital",
        "cash",
        "shares_outstanding",
        "market_cap",
        "fred_snapshot",
        "raw_text_id",
    }

    assert set(context_schema.IngestOutput.__annotations__.keys()) == expected_fields


def test_analysis_to_risk_required_values():
    assert context_schema.ANALYSIS_TO_RISK_REQUIRED == [
        "ratios",
        "anomalies",
        "ticker",
        "year",
    ]


def test_risk_to_report_required_values():
    assert context_schema.RISK_TO_REPORT_REQUIRED == [
        "z_score",
        "ml_risk_prob",
        "risk_flags",
        "peer_percentile",
    ]


def test_risk_output_has_required_fields():
    expected_fields = {
        "z_score",
        "ml_risk_prob",
        "risk_flags",
        "peer_percentile",
    }

    assert expected_fields.issubset(set(context_schema.RiskOutput.__annotations__.keys()))
