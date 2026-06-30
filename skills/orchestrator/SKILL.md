# OrchestratorAgent Skill

## Purpose
Route a natural-language bank analysis query through the full FinSight pipeline,
coordinating IngestAgent → AnalysisAgent → RiskAgent → NarrativeAgent → ReportAgent
in sequence. Never let agents call each other directly. Return a structured result dict.

## Input Schema
```
query: str  — e.g. "Analyze PNC 2023"
```
Validate with `_parse_query` before touching any agent:
- ticker: non-empty alphabetic string (1–6 chars), uppercased
- year: 4-digit int, not in future, not before 1993

## Output Contract
Success:
```json
{"status": "complete", "report_path": "<path>.pdf", "risk": {...}, "run_id": "<uuid>"}
```
Failure (any AgentExecutionError):
```json
{"status": "failed", "error": "<message>", "run_id": "<uuid>"}
```
Always returns a dict — never raises.

## Step-by-Step Behavior
1. Generate `run_id = str(uuid.uuid4())`.
2. Call `_parse_query(query)` → `{ticker, year}`. Return failed dict on ValueError.
3. Build `BASE_CONTEXT = {ticker, year, run_id, warnings: []}`.
4. For each agent in pipeline order:
   a. Instantiate the agent class.
   b. Call `agent.run(context)` — the agent writes its output under its own key.
   c. Catch `AgentExecutionError` → return `{status: failed, error: str, run_id}`.
5. Return `{status: complete, report_path, risk, run_id}`.

## Error Handling
- `ValueError` from `_parse_query` → `{status: failed, error: ..., run_id}`.
- `AgentExecutionError` from any pipeline agent → same failed dict.
- Never re-raise. Never let exceptions escape `run()`.

## Forbidden Behaviors
- Agents must not call each other directly.
- Do not swallow non-AgentExecutionError exceptions silently.
- Do not hardcode ticker or year values.

## Worked Example — PNC 2023
```
query = "Analyze PNC 2023"
→ _parse_query → {ticker: "PNC", year: 2023}
→ IngestAgent.run(ctx) → ctx["ingest"] populated
→ AnalysisAgent.run(ctx) → ctx["analysis"] populated
→ RiskAgent.run(ctx) → ctx["risk"] = {z_score: 0.253, ...}
→ NarrativeAgent.run(ctx) → ctx["narrative"] populated
→ ReportAgent.run(ctx) → ctx["report"] = {report_path: "reports/PNC_2023_report.pdf"}
→ return {status: "complete", report_path: "reports/PNC_2023_report.pdf",
          risk: {z_score: 0.253, ...}, run_id: "<uuid>"}
```
