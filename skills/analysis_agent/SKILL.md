# AnalysisAgent Skill

## Purpose
Compute financial ratios, detect anomalies, and generate charts from the ingest context produced by IngestAgent. Write structured output under `context["analysis"]`. Never call other agents. Never make LLM calls.

---

## Tools Available

```python
calc_ratios(ingest: dict) -> dict
    # Returns {"data": Ratios, "error": str | None}
    # Ratios keys: current_ratio (None for banks), debt_to_equity,
    #   return_on_assets, net_margin, interest_coverage
    # interest_expense floored at 0.001 to prevent division by zero

detect_anomalies(ratios: dict, fred_snapshot: dict) -> dict
    # Returns {"data": List[Anomaly], "error": str | None}
    # Flags: interest_coverage < 1.5 (high severity), T10Y2Y < 0 (yield inversion),
    #   UNRATE rising, net_margin < 0 (negative profitability)
    # Each Anomaly: {"metric": str, "description": str, "severity": str, "code": str}

plot_chart(data: dict, chart_type: str, output_path: str) -> dict
    # Returns {"data": {"path": str}, "error": str | None}
    # chart_type: "bar", "line", "heatmap"
    # Saves PNG to reports/charts/; uses Agg backend (headless)
```

---

## Input Schema

Receives `context` dict containing at minimum:
```python
{
    "ticker": str,
    "year": int,
    "run_id": str,
    "ingest": {
        "revenue": float, "net_income": float, "total_assets": float,
        "total_liabilities": float, "shareholders_equity": float,
        "ebit": float, "interest_expense": float,
        "current_assets": float | None, "current_liabilities": float | None,
        "fred_snapshot": dict,
        ...
    }
}
```

ANALYSIS_TO_RISK_REQUIRED keys must be present in output: `ratios`, `anomalies`, `ticker`, `year`.

---

## Output Contract

Writes results under `context["analysis"]`. Every key must be present.

```python
context["analysis"] = {
    "ratios": {
        "current_ratio": float | None,  # None for banks
        "debt_to_equity": float,
        "return_on_assets": float,
        "net_margin": float,
        "interest_coverage": float,
    },
    "anomalies": [
        {"metric": str, "description": str, "severity": str, "code": str},
        ...
    ],
    "chart_paths": [str, ...],   # paths to saved PNG files; [] if none generated
}
```

---

## Step-by-Step Behavior

1. Validate that `context.get("ingest")` exists and contains required keys (`revenue`, `net_income`, `total_assets`). Raise `AgentExecutionError` if missing.
2. Call `calc_ratios(context["ingest"])`. Check `result["error"]`. On error, raise `AgentExecutionError`.
3. Call `detect_anomalies(ratios, context["ingest"].get("fred_snapshot", {}))`. Check `result["error"]`. On error, log and continue with `anomalies = []`.
4. Call `plot_chart(...)` for each chart to generate. Save paths. On chart error, log and continue — chart generation failure must not abort the pipeline.
5. Assemble and write `context["analysis"]`.
6. Validate ANALYSIS_TO_RISK_REQUIRED keys are present. Raise `AgentExecutionError` if missing.

---

## Error Handling

- `calc_ratios` error → raise `AgentExecutionError` (ratios are required for downstream).
- `detect_anomalies` error → log warning, set `anomalies = []`, continue.
- `plot_chart` error → log warning, skip that chart, continue.
- Missing ingest keys → raise `AgentExecutionError` with clear message.

---

## Forbidden Behaviors

- Do not call RiskAgent, NarrativeAgent, or any other agent.
- Do not overwrite keys set by IngestAgent under `context["ingest"]`.
- Do not use `print()` — use the structured logger.
- Do not assume `current_ratio` is non-None — banks always return None.
- Do not access `result["data"]` without first confirming `result["error"] is None`.

---

## Worked Example — PNC 2023

Input ratios from fixture values:
- `debt_to_equity` = 504 / 53 = 9.509
- `return_on_assets` = 5.4 / 557 = 0.0097
- `net_margin` = 5.4 / 22 = 0.245
- `interest_coverage` = 6.1 / 4.2 = 1.452  ← triggers high-severity anomaly (< 1.5)
- `current_ratio` = None (bank, unclassified balance sheet)

Expected anomalies:
```python
[
    {
        "metric": "interest_coverage",
        "description": "Interest coverage ratio 1.452 is below 1.5 — earnings barely cover interest expense",
        "severity": "high",
        "code": "low_interest_coverage",
    },
    {
        "metric": "T10Y2Y",
        "description": "Yield curve inverted (T10Y2Y = -0.35) — historically precedes recession",
        "severity": "high",
        "code": "yield_curve_inverted",
    },
]
```
