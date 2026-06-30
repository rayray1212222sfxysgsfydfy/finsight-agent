# NarrativeAgent Skill

## Purpose
Generate a structured analyst narrative from the pipeline context (ingest + analysis + risk). Produce prose sections suitable for the final PDF report. This is the only agent that makes LLM calls. Write output under `context["narrative"]`.

---

## Tools Available

NarrativeAgent drives the Claude API directly (agentic loop). No tool functions from `src/tools/` are called — this agent synthesizes context into language.

```python
# Claude API agentic loop (see CLAUDE.md "Claude API" section):
client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=2000,
    system=load_skill("narrative_agent"),
    messages=[...],
)
# Loop until stop_reason != "tool_use" (NarrativeAgent typically stops after
# one turn since it has no tools to invoke — but always implement the loop).
```

---

## Input Schema

Receives full `context` dict:
```python
{
    "ticker": str,
    "year": int,
    "run_id": str,
    "ingest": { ... },     # IngestOutput
    "analysis": {
        "ratios": { ... },
        "anomalies": [...],
        "chart_paths": [...],
    },
    "risk": {
        "z_score": float,
        "z_zone": str,
        "ml_risk_prob": float,
        "ml_risk_label": str,
        "risk_flags": [...],
        "peer_percentile": float,
    }
}
```

---

## Output Contract

Writes results under `context["narrative"]`. Every key must be present — empty string is allowed, None is not.

```python
context["narrative"] = {
    "executive_summary": str,     # 2–3 sentence high-level overview
    "financial_highlights": str,  # key ratio findings, revenue/income trends
    "risk_assessment": str,       # Z-Score interpretation, ML label, flags
    "macro_context": str,         # FEDFUNDS, T10Y2Y, UNRATE impact on the bank
    "conclusion": str,            # forward-looking analyst judgment
}
```

---

## Step-by-Step Behavior

1. Validate `context.get("risk")` and `context.get("analysis")` exist. Raise `AgentExecutionError` if missing.
2. Build a structured prompt that injects all numeric context (ratios, Z-Score, risk flags, macro snapshot) as a JSON block. Instruct the model to produce exactly five prose sections.
3. Call the Claude API. Use `model="claude-sonnet-4-6"`, `max_tokens=2000`.
4. Implement the agentic loop: keep looping while `response.stop_reason == "tool_use"` (NarrativeAgent has no tools, so this loop typically runs once).
5. Parse the final text response into the five sections. Split on section headers if the model uses them, or extract via JSON if prompted in structured output mode.
6. Validate all five keys are non-empty strings. Raise `AgentExecutionError` if any are missing or empty.
7. Write `context["narrative"]`.

---

## Prompt Guidelines

Include in the system/user prompt:
- Ticker, year, and that this is a US commercial bank (contextualize low Z-Scores vs peers).
- All ratio values with labels.
- Z-Score value, zone, and peer percentile (e.g. "PNC's Z=0.253 ranks at the 60th percentile among peer banks — not a distress signal in this sector").
- All risk flags with severity.
- FRED macro snapshot values.
- Instruction: "Do not hallucinate values. Use only the numbers provided. Flag any missing data explicitly."

---

## Error Handling

- API error → raise `AgentExecutionError(f"NarrativeAgent: LLM call failed: {e}")`.
- Empty or malformed response → raise `AgentExecutionError("NarrativeAgent: response missing expected sections")`.
- Missing upstream context → raise `AgentExecutionError` with clear message.

---

## Forbidden Behaviors

- Do not invent or extrapolate numbers not present in the context.
- Do not call other agents.
- Do not overwrite `context["ingest"]`, `context["analysis"]`, or `context["risk"]`.
- Do not use `print()` — use the structured logger.
- Zero tolerance for hallucinated financial figures — LLM-as-judge eval will score hallucination separately.

---

## Worked Example — PNC 2023

Given:
- Revenue $22B, Net Income $5.4B, Net Margin 24.5%
- Interest Coverage 1.452 (high-severity anomaly)
- Z-Score 0.253, zone "distress", peer_percentile 0.55
- T10Y2Y -0.35 (yield curve inverted), FEDFUNDS 5.33

Expected `executive_summary` (illustrative):
> "PNC Financial Services reported $22B in revenue and $5.4B in net income for fiscal year 2023, reflecting a net margin of 24.5%. While the Altman Z-Score of 0.253 falls in the distress zone under absolute thresholds, this is consistent with the highly leveraged balance sheet structure common to large US commercial banks; PNC ranks at the 55th percentile among sector peers. Key risks include an interest coverage ratio of 1.45 and a persistently inverted yield curve."
