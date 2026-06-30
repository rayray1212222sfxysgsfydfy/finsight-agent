# RiskAgent Skill

## Purpose
Score credit risk for a bank using Altman Z-Score, an ML classifier, and rule-based risk flags. Compare the bank's Z-Score to sector peers. Write structured output under `context["risk"]`. Never call other agents. Never make LLM calls.

---

## Tools Available

```python
score_altman_z(ingest: dict) -> dict
    # Returns {"data": {"z_score": float, "zone": str}, "error": str | None}
    # zone: "safe" (>2.99), "grey" (1.81-2.99), "distress" (<1.81)
    # WC/TA term uses WC=0 when working_capital is None (standard for banks)
    # Formula: Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA)
    #              + 0.6*(MktCap/TL) + 1.0*(Rev/TA)

run_ml_risk_model(features: dict) -> dict
    # Returns {"data": {"probability": float, "label": str}, "error": str | None}
    # features must match feature_names.json order (7 keys, no altman_z):
    #   debt_to_equity, return_on_assets, interest_coverage,
    #   revenue_growth, net_margin, fedfunds, t10y2y
    # label: "low", "moderate", "high"
    # Loads risk_classifier.pkl (combined scaler+RF pipeline)

flag_risks(z_score: float, ratios: dict, fred_snapshot: dict) -> dict
    # Returns {"data": List[RiskFlag], "error": str | None}
    # Always flags: T10Y2Y < 0 (yield curve inversion), rising UNRATE
    # Also flags: z_score < 1.81 (with peer context), interest_coverage < 1.5
    # RiskFlag: {"code": str, "description": str, "severity": str}

compare_to_peers(z_score: float, peer_z_scores: list) -> dict
    # Returns {"data": float, "error": str | None}
    # data = fraction of peers at-or-below this z_score (0.0 to 1.0)
    # Higher = relatively safer vs peers
```

---

## Input Schema

Receives `context` dict containing at minimum:
```python
{
    "ticker": str,
    "year": int,
    "run_id": str,
    "ingest": { ... },       # full IngestOutput
    "analysis": {
        "ratios": { ... },   # Ratios dict
        "anomalies": [...],
        "chart_paths": [...],
    }
}
```

RISK_TO_REPORT_REQUIRED keys must be present in output: `z_score`, `ml_risk_prob`, `risk_flags`, `peer_percentile`.

---

## Output Contract

Writes results under `context["risk"]`. Every key must be present.

```python
context["risk"] = {
    "z_score": float,
    "z_zone": str,             # "safe", "grey", or "distress"
    "ml_risk_prob": float,     # 0.0–1.0 probability of high-risk class
    "ml_risk_label": str,      # "low", "moderate", or "high"
    "risk_flags": [
        {"code": str, "description": str, "severity": str},
        ...
    ],
    "peer_percentile": float,  # 0.0–1.0; fraction of peers at-or-below this Z
}
```

---

## Step-by-Step Behavior

1. Validate `context.get("ingest")` and `context.get("analysis", {}).get("ratios")` exist. Raise `AgentExecutionError` if missing.
2. Call `score_altman_z(context["ingest"])`. Check `result["error"]`. On error, raise `AgentExecutionError` (Z-Score is required).
3. Build the ML feature dict from `context["analysis"]["ratios"]` and `context["ingest"]["fred_snapshot"]`. Required keys: `debt_to_equity`, `return_on_assets`, `interest_coverage`, `revenue_growth`, `net_margin`, `fedfunds`, `t10y2y`. Note: `revenue_growth` may need to be computed here if not in ratios.
4. Call `run_ml_risk_model(features)`. Check `result["error"]`. On error, log and set `ml_risk_prob = 0.0`, `ml_risk_label = "unknown"` (non-fatal — Z-Score is the primary signal).
5. Call `flag_risks(z_score, ratios, fred_snapshot)`. Check `result["error"]`. On error, set `risk_flags = []`.
6. Call `compare_to_peers(z_score, peer_z_scores)`. `peer_z_scores` should be loaded from prior context or a known sector benchmark list. On error, set `peer_percentile = 0.5` (neutral default).
7. Assemble and write `context["risk"]`.
8. Validate RISK_TO_REPORT_REQUIRED keys. Raise `AgentExecutionError` if missing.

---

## Error Handling

- `score_altman_z` error → raise `AgentExecutionError` (required field).
- `run_ml_risk_model` error → log, set `ml_risk_prob = 0.0`, `ml_risk_label = "unknown"`, continue.
- `flag_risks` error → log, set `risk_flags = []`, continue.
- `compare_to_peers` error → log, set `peer_percentile = 0.5`, continue.

---

## Forbidden Behaviors

- Do not interpret a bank Z-Score < 1.81 as automatically distressed without peer context — all banks in this dataset score below 1.81 (structural leverage). Always contextualize via `peer_percentile`.
- Do not load `.pkl` files directly — only call `run_ml_risk_model()`.
- Do not call other agents.
- Do not overwrite `context["ingest"]` or `context["analysis"]`.

---

## Worked Example — PNC 2023

From fixture values (revenue=22B, net_income=5.4B, total_assets=557B, total_liabilities=504B,
shareholders_equity=53B, ebit=6.1B, interest_expense=4.2B, retained_earnings=29B,
market_cap=~55B, working_capital=None):

```python
# Altman Z (WC term = 0 for banks):
# Z = 0 + 1.4*(29/557) + 3.3*(6.1/557) + 0.6*(55/504) + 1.0*(22/557)
# Z ≈ 0.253  →  zone = "distress"  (expected for leveraged banks)

context["risk"] = {
    "z_score": 0.253,
    "z_zone": "distress",
    "ml_risk_prob": <model output>,
    "ml_risk_label": <"low" | "moderate" | "high">,
    "risk_flags": [
        {"code": "yield_curve_inverted", "description": "T10Y2Y = -0.35", "severity": "high"},
        {"code": "low_interest_coverage", "description": "Coverage 1.452 < 1.5", "severity": "high"},
    ],
    "peer_percentile": <fraction of peer Z-Scores ≤ 0.253>,
}
```
