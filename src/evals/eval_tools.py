"""Tier 1 — Tool unit evals.

Validates that every tool function honours the error-handling contract
(always returns a dict with 'data' and 'error' keys) and that core
computations are correct on known inputs.  No real API calls — all I/O
is mocked.
"""

from src.tools.analysis_tools import calc_ratios, detect_anomalies
from src.tools.report_tools import format_table, generate_pdf
from src.tools.risk_tools import compare_to_peers, flag_risks, score_altman_z


# ---------------------------------------------------------------------------
# Contract helpers
# ---------------------------------------------------------------------------

def _is_valid_result(result: dict) -> bool:
    """Return True if result has exactly 'data' and 'error' keys."""
    return isinstance(result, dict) and "data" in result and "error" in result


# ---------------------------------------------------------------------------
# calc_ratios
# ---------------------------------------------------------------------------

def _base_ingest() -> dict:
    return {
        "revenue": 22e9,
        "net_income": 5.4e9,
        "total_assets": 557e9,
        "total_liabilities": 504e9,
        "shareholders_equity": 53e9,
        "ebit": 6.1e9,
        "interest_expense": 4.2e9,
        "retained_earnings": 29e9,
        "cash": 31e9,
        "market_cap": 55e9,
        "current_assets": None,
        "current_liabilities": None,
        "working_capital": None,
    }


def eval_calc_ratios_returns_valid_contract():
    result = calc_ratios(_base_ingest())
    assert _is_valid_result(result), "calc_ratios must return data/error dict"
    assert result["error"] is None
    assert result["data"] is not None


def eval_calc_ratios_net_margin_correct():
    result = calc_ratios(_base_ingest())
    nm = result["data"]["net_margin"]
    expected = 5.4e9 / 22e9
    assert abs(nm - expected) < 1e-6, f"net_margin {nm} != {expected}"


def eval_calc_ratios_debt_to_equity_correct():
    result = calc_ratios(_base_ingest())
    dte = result["data"]["debt_to_equity"]
    expected = 504e9 / 53e9
    assert abs(dte - expected) < 1e-4, f"debt_to_equity {dte} != {expected}"


def eval_calc_ratios_missing_required_field_returns_error():
    ingest = _base_ingest()
    del ingest["total_assets"]
    result = calc_ratios(ingest)
    assert result["error"] is not None, "Missing required field must produce error"
    assert result["data"] is None


def eval_calc_ratios_current_ratio_is_none_for_banks():
    result = calc_ratios(_base_ingest())
    assert result["data"]["current_ratio"] is None, "Banks have no current_ratio"


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------

def eval_detect_anomalies_returns_valid_contract():
    ratios = calc_ratios(_base_ingest())["data"]
    fred = {"FEDFUNDS": 5.33, "T10Y2Y": -0.35, "UNRATE": 3.7}
    result = detect_anomalies(ratios, fred)
    assert _is_valid_result(result)
    assert result["error"] is None
    assert isinstance(result["data"], list)


def eval_detect_anomalies_flags_low_interest_coverage():
    ingest = _base_ingest()
    ingest["interest_expense"] = ingest["ebit"] * 2  # coverage < 1
    ratios = calc_ratios(ingest)["data"]
    result = detect_anomalies(ratios, {})
    codes = [a["code"] for a in result["data"]]
    assert "low_interest_coverage" in codes


# ---------------------------------------------------------------------------
# score_altman_z
# ---------------------------------------------------------------------------

def eval_score_altman_z_pnc_fixture():
    ingest = _base_ingest()
    ingest["working_capital"] = 18e9
    result = score_altman_z(ingest)
    assert result["error"] is None
    assert abs(result["data"]["z_score"] - 0.253) < 0.01
    assert result["data"]["z_zone"] == "distress"


def eval_score_altman_z_missing_field_returns_error():
    ingest = _base_ingest()
    del ingest["market_cap"]
    result = score_altman_z(ingest)
    assert result["error"] is not None
    assert result["data"] is None


# ---------------------------------------------------------------------------
# flag_risks
# ---------------------------------------------------------------------------

def eval_flag_risks_yield_curve_inversion():
    ratios = calc_ratios(_base_ingest())["data"]
    fred = {"T10Y2Y": -0.35, "FEDFUNDS": 5.33, "UNRATE": 3.7}
    result = flag_risks(ratios, fred)
    assert result["error"] is None
    codes = [f["code"] for f in result["data"]]
    assert "yield_curve_inverted" in codes


def eval_flag_risks_empty_fred_no_crash():
    ratios = calc_ratios(_base_ingest())["data"]
    result = flag_risks(ratios, {})
    assert _is_valid_result(result)
    assert result["error"] is None


# ---------------------------------------------------------------------------
# compare_to_peers
# ---------------------------------------------------------------------------

def eval_compare_to_peers_percentile_correct():
    result = compare_to_peers(0.28, [0.20, 0.25, 0.28, 0.30])
    assert result["error"] is None
    assert result["data"] == 0.75  # 3 out of 4 at-or-below


def eval_compare_to_peers_empty_list_returns_error():
    result = compare_to_peers(0.28, [])
    assert result["error"] is not None


# ---------------------------------------------------------------------------
# format_table / generate_pdf
# ---------------------------------------------------------------------------

def eval_format_table_returns_valid_contract():
    rows = [{"A": "x", "B": "y"}]
    result = format_table(rows, columns=["A", "B"])
    assert _is_valid_result(result)
    assert result["error"] is None
    assert "table" in result["data"]


def eval_generate_pdf_bad_extension_returns_error():
    result = generate_pdf("out.txt", "PNC", 2023, "narrative", "table")
    assert result["error"] is not None
    assert result["data"] is None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_EVALS = [
    eval_calc_ratios_returns_valid_contract,
    eval_calc_ratios_net_margin_correct,
    eval_calc_ratios_debt_to_equity_correct,
    eval_calc_ratios_missing_required_field_returns_error,
    eval_calc_ratios_current_ratio_is_none_for_banks,
    eval_detect_anomalies_returns_valid_contract,
    eval_detect_anomalies_flags_low_interest_coverage,
    eval_score_altman_z_pnc_fixture,
    eval_score_altman_z_missing_field_returns_error,
    eval_flag_risks_yield_curve_inversion,
    eval_flag_risks_empty_fred_no_crash,
    eval_compare_to_peers_percentile_correct,
    eval_compare_to_peers_empty_list_returns_error,
    eval_format_table_returns_valid_contract,
    eval_generate_pdf_bad_extension_returns_error,
]


def run_tool_evals() -> dict:
    """Run all tool evals and return a summary dict."""
    passed, failed = [], []
    for fn in _EVALS:
        try:
            fn()
            passed.append(fn.__name__)
        except Exception as exc:
            failed.append((fn.__name__, str(exc)))
    return {"passed": passed, "failed": failed, "total": len(_EVALS)}


if __name__ == "__main__":
    summary = run_tool_evals()
    print(f"Tool evals: {len(summary['passed'])}/{summary['total']} passed")
    for name, err in summary["failed"]:
        print(f"  FAIL {name}: {err}")
