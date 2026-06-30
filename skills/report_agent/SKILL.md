# ReportAgent Skill

## Purpose
Assemble all pipeline outputs (narrative sections, charts, tables) into a formatted PDF analyst report. Save the PDF to `reports/`. Write the output path under `context["report"]`. Never call other agents. Never make LLM calls.

---

## Tools Available

```python
generate_pdf(content: dict, output_path: str) -> dict
    # Returns {"data": {"path": str}, "error": str | None}
    # content keys: title, sections (list of {heading, body}),
    #   tables (list of dicts), chart_paths (list of PNG paths)
    # Saves a formatted PDF to the given output_path

format_table(data: dict, columns: list) -> dict
    # Returns {"data": {"rows": list, "headers": list}, "error": str | None}
    # Converts a flat dict or list of dicts into a structured table for generate_pdf
```

---

## Input Schema

Receives full `context` dict:
```python
{
    "ticker": str,
    "year": int,
    "run_id": str,
    "ingest": { ... },
    "analysis": {
        "ratios": { ... },
        "anomalies": [...],
        "chart_paths": [...],
    },
    "risk": {
        "z_score": float, "z_zone": str,
        "ml_risk_prob": float, "ml_risk_label": str,
        "risk_flags": [...], "peer_percentile": float,
    },
    "narrative": {
        "executive_summary": str,
        "financial_highlights": str,
        "risk_assessment": str,
        "macro_context": str,
        "conclusion": str,
    }
}
```

---

## Output Contract

Writes result under `context["report"]`. Both keys must be present.

```python
context["report"] = {
    "pdf_path": str,     # absolute path to saved PDF, e.g. "reports/PNC_2023_<run_id>.pdf"
    "status": str,       # "success" or "error"
}
```

---

## Step-by-Step Behavior

1. Validate all required upstream keys: `context["narrative"]`, `context["risk"]`, `context["analysis"]`. Raise `AgentExecutionError` if missing.
2. Build the title string: `f"{ticker} Financial Intelligence Report — FY{year}"`.
3. Call `format_table(context["analysis"]["ratios"], columns=["Metric", "Value"])` to produce the ratios table.
4. Call `format_table` on `context["risk"]["risk_flags"]` to produce the risk flags table.
5. Assemble `content` dict:
   - `title`: report title
   - `sections`: list of `{"heading": str, "body": str}` from `context["narrative"]` (executive_summary, financial_highlights, risk_assessment, macro_context, conclusion — in that order)
   - `tables`: [ratios_table, risk_flags_table]
   - `chart_paths`: `context["analysis"]["chart_paths"]`
6. Build `output_path = f"reports/{ticker}_{year}_{run_id[:8]}.pdf"`. Ensure `reports/` directory exists.
7. Call `generate_pdf(content, output_path)`. Check `result["error"]`. On error, raise `AgentExecutionError`.
8. Write `context["report"] = {"pdf_path": result["data"]["path"], "status": "success"}`.

---

## Error Handling

- `format_table` error → log and skip that table (PDF can still be generated without it).
- `generate_pdf` error → raise `AgentExecutionError(f"ReportAgent: PDF generation failed: {error}")`.
- Missing upstream context → raise `AgentExecutionError` with clear message identifying which agent's output is absent.

---

## Forbidden Behaviors

- Do not call other agents.
- Do not overwrite any upstream context keys.
- Do not use `print()` — use the structured logger.
- Do not hardcode output paths — always derive from `ticker`, `year`, `run_id`.

---

## Worked Example — PNC 2023

Expected output:
```python
context["report"] = {
    "pdf_path": "reports/PNC_2023_abc12345.pdf",
    "status": "success",
}
```

PDF sections in order:
1. Executive Summary
2. Financial Highlights (with ratios table)
3. Risk Assessment (with risk flags table, Z-Score = 0.253, peer_percentile = 0.55)
4. Macro Context (FEDFUNDS 5.33, T10Y2Y -0.35 inverted)
5. Conclusion
6. Charts (appended from chart_paths)
