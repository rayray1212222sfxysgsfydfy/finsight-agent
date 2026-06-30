# OrchestratorAgent Skill

## Purpose
Parse the user query, validate inputs, route context through the full agent pipeline (Ingest → Analysis → Risk → Narrative → Report), and return the final PDF path. The Orchestrator is the only entry point — agents never call each other directly.

---

## Tools Available

OrchestratorAgent does not call `src/tools/` functions directly. It instantiates and calls each agent in sequence, passing the shared `context` dict.

```python
# Agents called in order (never call agents directly from other agents):
IngestAgent(context).run()
AnalysisAgent(context).run()
RiskAgent(context).run()
NarrativeAgent(context).run()
ReportAgent(context).run()
```

---

## Input Schema

Receives a natural language query string:
```python
query: str   # e.g. "Analyze PNC 2023"
```

Optionally receives:
```python
track_cost: bool = False   # if True, print token + $ summary after run
```

---

## Output

Returns the path to the generated PDF as a string:
```python
"reports/PNC_2023_<run_id>.pdf"
```

Also prints the PDF path to stdout on success.

---

## Query Parsing — `_parse_query(query: str)`

Must extract `ticker` and `year` from the query string.

**Validation rules (enforce strictly before any agent is called):**
- `ticker`: non-empty string, uppercase, alphanumeric only. Raise `ValueError("Ticker must be a non-empty alphanumeric string")` if blank or invalid.
- `year`: 4-digit integer. Raise `ValueError("Year must be a 4-digit integer")` if not 4 digits.
- `year` must not be in the future (> current year). Raise `ValueError(f"Year {year} is in the future")`.
- `year` must not be before 1993. Raise `ValueError(f"Year {year} is before EDGAR history (1993)")`.

Never let bad input reach a tool function.

---

## Step-by-Step Behavior

1. Call `_parse_query(query)` → `(ticker, year)`. Any `ValueError` propagates directly to the caller (no wrapping).
2. Generate a unique `run_id` (e.g. `uuid4().hex`).
3. Initialise `context = {"ticker": ticker, "year": year, "run_id": run_id}`.
4. Log run start via structured logger.
5. Run each agent in sequence. Catch `AgentExecutionError` from each; log and re-raise so the CLI can report clearly.
6. After all agents complete, return `context["report"]["pdf_path"]`.
7. If `track_cost=True`, print token usage and estimated cost (read from response metadata where available).

---

## Error Handling

- `ValueError` from `_parse_query` → let propagate (caller's responsibility to show the user).
- `AgentExecutionError` from any agent → log with `run_id` and re-raise. Do not silently swallow.
- Unexpected exceptions → wrap in `AgentExecutionError` with `run_id` context before raising.

---

## Forbidden Behaviors

- Do not let agents call each other — all routing is through the Orchestrator.
- Do not pass partial or unvalidated tickers/years to any agent.
- Do not use `print()` for logging — use the structured logger (CLI output for the PDF path is acceptable).
- Do not hardcode model names — use `config.MODEL` (which resolves to `claude-sonnet-4-6`).

---

## Worked Example

```bash
python run.py "Analyze PNC 2023"
```

1. `_parse_query("Analyze PNC 2023")` → `ticker="PNC"`, `year=2023`
2. `run_id = "abc12345..."`
3. `context = {"ticker": "PNC", "year": 2023, "run_id": "abc12345..."}`
4. IngestAgent → writes `context["ingest"]`
5. AnalysisAgent → writes `context["analysis"]`
6. RiskAgent → writes `context["risk"]`
7. NarrativeAgent → writes `context["narrative"]`
8. ReportAgent → writes `context["report"]`
9. Returns and prints: `"reports/PNC_2023_abc12345.pdf"`

**Invalid input examples that must raise ValueError before any agent runs:**
```python
"Analyze  2023"        # empty ticker
"Analyze PNC 20230"    # 5-digit year
"Analyze PNC 2030"     # future year
"Analyze PNC 1985"     # before 1993
```
