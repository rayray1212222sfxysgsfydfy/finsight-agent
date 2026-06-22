# FinSight — Claude Code Instructions

You are building FinSight, a multi-agent financial intelligence system for DS 2000 at UPitt.
Read this entire file before writing any code. Append every bug, decision, and API quirk
you discover to the Session Log at the bottom — that log is your memory across sessions.

---

## What You Are Building

CLI takes a bank ticker + year and produces a PDF analyst report autonomously. Pipeline:
`IngestAgent` → `AnalysisAgent` → `RiskAgent` → `NarrativeAgent` → `ReportAgent`, all
routed by `OrchestratorAgent`. Agents never call each other directly.

```bash
python run.py "Analyze PNC 2023"
python run.py "Analyze PNC 2023" --track-cost    # bonus: print token + $ summary
```

Validate in `OrchestratorAgent._parse_query()` before touching any agent: ticker must be
a non-empty string, year must be 4-digit int not in future and not before 1993. Raise
`ValueError` with a clear message. Never let bad input reach a tool function.

**Model:** `claude-sonnet-4-6` everywhere. Do not substitute.
**Focus banks:** PNC Financial, JPMorgan Chase, BNY Mellon + 2 TBD.

---

## Repo Layout

```
finsight/
├── run.py
├── CLAUDE.md
├── requirements.txt              # update in same commit as any new import
├── .env.example                  # never commit .env
├── src/
│   ├── agents/
│   │   ├── orchestrator.py       OrchestratorAgent
│   │   ├── ingest_agent.py       IngestAgent
│   │   ├── analysis_agent.py     AnalysisAgent
│   │   ├── risk_agent.py         RiskAgent
│   │   ├── narrative_agent.py    NarrativeAgent
│   │   ├── report_agent.py       ReportAgent
│   │   └── context_schema.py     write this before any agent — all agents depend on it
│   ├── tools/                    pure functions only, no LLM calls, all unit-tested
│   │   ├── sec_tools.py          fetch_10k_filing
│   │   ├── fred_tools.py         fetch_fred_series, get_macro_snapshot
│   │   ├── market_tools.py       fetch_market_data
│   │   ├── db_tools.py           write_to_db, read_from_db, check_cache
│   │   ├── analysis_tools.py     calc_ratios, detect_anomalies, plot_chart
│   │   ├── risk_tools.py         score_altman_z, run_ml_risk_model, flag_risks, compare_to_peers
│   │   ├── rag_tools.py          embed_text, query_vector_store
│   │   └── report_tools.py       generate_pdf, format_table
│   ├── models/
│   │   ├── train.py
│   │   ├── risk_classifier.pkl   gitignore if > 10MB
│   │   ├── feature_pipeline.pkl
│   │   └── feature_names.json    ordered feature list — source of truth for model input
│   ├── evals/
│   │   ├── eval_tools.py         tier 1
│   │   ├── eval_agents.py        tier 2
│   │   ├── eval_pipeline.py      tier 2
│   │   └── eval_llm_judge.py     tier 3
│   └── utils/
│       ├── logger.py             structured logger → SQLite agent_traces table
│       ├── config.py             all env var access goes here only
│       └── exceptions.py         AgentExecutionError
├── skills/                       one SKILL.md per agent, loaded as system prompt at runtime
│   └── <agent_name>/SKILL.md
├── tests/                        mirrors src/ exactly — test_<source_filename>.py
│   └── conftest.py               ALL shared fixtures here, nowhere else
├── data/raw/                     gitignored
├── data/processed/
├── data/finsight.db              gitignored
├── reports/                      gitignored
├── logs/                         gitignored
└── notebooks/eda.ipynb           week 2 — 5+ charts, 5 banks, 2018–2023
```

---

## Conventions

**Naming:** functions `verb_noun`, classes `NounAgent`/`NounTool`, constants `ALL_CAPS`,
tests `test_[fn]_[condition]_[expected]`, branches `type/scope-desc`.

**Every public function:** type hints on all params + return, single-line docstring.

**Imports:** stdlib → third-party → local. No circular imports. Tools never import from
agents. Agents never import from each other. All env vars through `config.py` only.

---

## Error Handling Contract

Tool functions (`src/tools/`) **never raise**. Always return:
```python
{"data": <result>, "error": None}                          # success
{"data": None,     "error": "what failed and why"}         # any failure
```
Only permitted exception: `raise ValueError("clear message")` for invalid inputs at the top
of the function, before any I/O.

Agent classes **may raise** `AgentExecutionError` for unrecoverable failures.

**Always check `result["error"]` before accessing `result["data"]`.** Never chain
`result["data"]["field"]` without confirming error is None first.

---

## Agent Context Schema

Define in `context_schema.py` before writing any agent.

```python
# orchestrator passes to every agent
BASE_CONTEXT = {"ticker": str, "year": int, "run_id": str}

# IngestAgent adds under context["ingest"]
INGEST_OUTPUT = {
    "revenue": float, "net_income": float, "total_assets": float,
    "total_liabilities": float, "shareholders_equity": float,
    "current_assets": float, "current_liabilities": float,
    "ebit": float, "interest_expense": float, "retained_earnings": float,
    "working_capital": float, "cash": float, "market_cap": float,
    "fred_snapshot": dict, "raw_text_id": str,
}
```

Always read context with `.get(key, default)`. Never assume a key exists. Never overwrite
a key set by a prior agent — add your output under your own key (e.g. `context["risk"]`).

---

## Tests

Write the test before the implementation. No exceptions.

- Test files mirror source: `src/tools/sec_tools.py` → `tests/test_sec_tools.py`
- All shared fixtures in `tests/conftest.py` — never duplicate across files
- Never make real API calls in tests — mock everything
- `pytest tests/ -v` must be 100% green before every commit

**Minimum counts (graded):** 10 tool unit tests, 5 per agent, 5 integration, 10 LLM-judge
samples, 5 failure/recovery tests. Total 35+.

**PNC 2023 fixture values (use these, they are verified):**
revenue=22B, net_income=5.4B, total_assets=557B, total_liabilities=504B,
shareholders_equity=53B, ebit=6.1B, interest_expense=4.2B, market_cap=55B,
retained_earnings=29B, working_capital=18B, cash=31B.

---

## Git

Branches: `feature/`, `fix/`, `eval/`, `refactor/`, `docs/` + `scope-description`.

Commits — Conventional Commits strictly:
```
feat(risk): add Altman Z-score computation tool
fix(ingest): handle EDGAR 429 with exponential backoff
test(evals): add LLM-as-judge narrative quality eval
```
Never commit: `.env`, `data/raw/`, `*.db`, `reports/`, `logs/`, `.pkl` > 10MB.
Read `git diff --staged` before every commit.

---

## Forbidden

```python
print(...)                          # use logger
except: pass                        # bare except forbidden
fetch_10k_filing("PNC", 2023)       # in a test — mock it
class Agent: results = []           # class-level state
client.messages.create(...)         # inside src/tools/
RiskAgent().run(...)                # agent calling agent directly
result["data"]["revenue"]           # without checking result["error"] first
api_key="sk-ant-..."                # hardcoded secret
import httpx                        # without adding to requirements.txt
```

---

## Data Sources

**EDGAR:** base `https://data.sec.gov/`. Rate limit 10 req/s. Every request needs header
`User-Agent: {config.SEC_USER_AGENT}` — missing header returns silent 403. Retry 429 with
`time.sleep(2 ** attempt)`, 3 attempts. XBRL field names vary by company/year — maintain
`FIELD_ALIASES` dict in `sec_tools.py` and try all variants before returning error.

**FRED:** `https://api.stlouisfed.org/fred/series/observations`. Auth via `api_key` param.
Series: `FEDFUNDS`, `CPIAUCSL`, `UNRATE`, `GS10`, `T10Y2Y`. Cache to SQLite. Use the last
observation on or before fiscal year end date — not today's values.

**yfinance:** no key needed. Always `.get("marketCap", None)` — returns None intermittently
even for valid tickers. Field names change without notice; patch in `market_tools.py`.

**ChromaDB:** `PersistentClient(path="data/chromadb/")`. Create that directory before
instantiating — it silently fails if missing. Default embedding function needs `onnxruntime`
in requirements.txt. Collection: `finsight_filings`. Chunks: 512 tokens, 50 overlap.

---

## ML Model

Features (order must match `feature_names.json`):
`debt_to_equity`, `return_on_assets`, `interest_coverage`,
`revenue_growth`, `net_margin`, `altman_z`, `fedfunds`, `t10y2y`.

- `current_ratio` was DROPPED from the feature set (and `working_capital`, which was
  only ever an ingest field feeding `altman_z`, is treated as Optional/None). Banks file
  unclassified balance sheets, so `AssetsCurrent`/`LiabilitiesCurrent` are absent from
  XBRL → both are structurally NaN for our entire 5-bank dataset. See sec_tools.py.
- `interest_coverage`: use `max(interest_expense, 0.001)` — division by zero for some banks
- `revenue_growth`: `pct_change()` produces NaN on first row — always `dropna()` before
  training and inference
- Any feature change requires full retrain + updated `feature_names.json`
- Minimum accuracy: 60% on holdout. Do not merge below this threshold.
- At runtime, call `run_ml_risk_model(features)` — never load `.pkl` directly from agents

---

## Altman Z-Score

```
Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MktCap/TL) + 1.0*(Rev/TA)
Zones: > 2.99 safe, 1.81–2.99 grey, < 1.81 distress
PNC 2023 verified: Z = 0.284 from real EDGAR data, Z = 0.253 from fixture values, zone = distress for both
(real Z uses HISTORICAL market cap = per-year shares × year-end close; earlier 0.328 used
yfinance current cap. WC term = 0 — banks file no classified working capital. Do NOT assert 1.847.)
```

Banks run structurally low Z-Scores due to high leverage. Always contextualize against
sector peers in `flag_risks` output — never interpret a bank Z < 1.81 as automatically
distressed. Flag `T10Y2Y < 0` (yield curve inversion) and rising `UNRATE` in every report.

---

## SKILL.md

Write before the agent class. Load as system prompt at runtime:
```python
def load_skill(agent_name: str) -> str:
    return Path(f"skills/{agent_name}/SKILL.md").read_text()
```

Required sections: Purpose, Tools Available (exact signatures), Input Schema, Output Contract
(every key always present — null is ok, missing is not), Step-by-Step Behavior, Error
Handling, Forbidden Behaviors, Worked Example using PNC 2023 values.

---

## Claude API

Tool use requires an agentic loop — a single call returns the tool invocation but never
executes it. Loop until `stop_reason != "tool_use"`:
```python
while response.stop_reason == "tool_use":
    tool_results = execute_tools(response.content)
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})
    response = client.messages.create(model="claude-sonnet-4-6", messages=messages, ...)
```
Use `max_tokens=2000` for agents with tools, `max_tokens=1000` for structured-output calls.

---

## Evals

| Type | Min tests | Threshold |
|---|---|---|
| Tool unit | 10 | 100% |
| Agent contract | 5 per agent | 100% |
| Pipeline integration | 5 | 80% |
| LLM-as-judge narrative | 10 | avg ≥ 3.5/5, hallucination = 5 |
| Failure/recovery | 5 | 100% graceful |

LLM judge scores accuracy, specificity, hallucination 1–5. Zero tolerance on hallucination.

---

## Week Gates

| Week | Done when |
|---|---|
| 1 | `pytest tests/ -v` 100% green, zero real API calls in tests |
| 2 | Model accuracy ≥ 60%, EDA notebook has 5+ charts for 5 banks 2018–2023 |
| 3 | `python run.py "Analyze PNC 2023"` produces a PDF end-to-end |
| 4 | 35+ tests across all eval tiers, demo runs live without error |

---

## Session Log

Append every bug, decision, API discovery, convention change, or retrain here.
Format: `[YYYY-MM-DD] CATEGORY: what happened and why it matters.`
Categories: `DECISION` `BUG` `CAVEAT` `FIX` `RETRAIN` `API` `CONVENTION`
[2026-06-15] 

CONVENTION: context_schema.py finalized — ANALYSIS_TO_RISK_REQUIRED uses
nested ratios/anomalies (not flattened feature names); RISK_TO_REPORT_REQUIRED separates
z_score (Altman) from ml_risk_prob (classifier). AnalysisOutput/RiskOutput/Ratios/Anomaly/
RiskFlag TypedDicts added to define internal shapes for downstream agents.

[2026-06-17] BUG: fred_tools tests had two fixture issues — test asserted wrong 
result["data"] shape (list not dict), and mock_get missing **kwargs for timeout 
argument. Fixed in tests only, implementation was correct.

[2026-06-18] BUG: market_tools tests used wrong mock path 
'src.tools.market_tools.yfinance.Ticker' — actual import alias 
is 'yf' so correct path is 'src.tools.market_tools.yf.Ticker'. 
Rule: mock where the name is used, not where it's defined.

[2026-06-18] CAVEAT: .venv was empty (only pip/setuptools) despite prior logs
referencing green tests — environment had been reset. Restored with
`.venv/bin/pip install -r requirements.txt`. If `pytest` reports "No module
named pytest", reinstall requirements before assuming a code problem.

[2026-06-18] CONVENTION: db_tools uses a module-level DB_PATH (defaults to
data/finsight.db) and a _get_connection() helper that auto-detects file: URIs
(uri=True). Tests monkeypatch DB_PATH to "file::memory:?cache=shared" and hold
a keeper connection open so separate per-call connections share the in-memory
DB. read_from_db treats a missing key as an error ("Key 'x' not found");
check_cache treats a miss as a non-error (data=None, error=None) so callers can
branch on cache misses.

[2026-06-20] FIX: fred_tools.fetch_fred_series omitted file_type=json, so FRED
returned XML and response.json() raised "Expecting value" — get_macro_snapshot
failed for every series. Added file_type=json param + a test asserting it. Real
get_macro_snapshot('2023-12-31') now works (FEDFUNDS=5.33, T10Y2Y=-0.35 inverted).

[2026-06-20] API: rewrote sec_tools.fetch_10k_filing. OLD impl hit
edgar/data/{ticker}/{year}/full-submission.txt — wrong: EDGAR needs numeric CIK
and that path 404s for every ticker. NEW impl: resolve ticker->CIK via
www.sec.gov/files/company_tickers.json (cached), then pull
data.sec.gov/api/xbrl/companyfacts/CIK##########.json. _select_annual_value
picks the annual (full-year/instant, 10-K-preferred) fact for the target year.
Module-level caches (_TICKER_MAP_CACHE/_FACTS_CACHE, _clear_caches() for tests)
so a 30 ticker-year loop downloads each payload once. All 5 focus banks return
real 2018-2023 data.

[2026-06-20] CAVEAT: bank XBRL field gaps (surfaced by EDA, now handled in
sec_tools). (1) Banks file UNCLASSIFIED balance sheets -> no AssetsCurrent/
LiabilitiesCurrent -> current_ratio & working_capital are None/NaN for all banks
(not fatal; only revenue/net_income/total_assets are REQUIRED_FIELDS). (2) No
OperatingIncomeLoss -> EBIT proxied by pre-tax income
(IncomeLossFromContinuingOperationsBeforeIncomeTaxes...). (3) Cash uses
CashAndDueFromBanks, not CashAndCashEquivalentsAtCarryingValue. (4) BK (BNY
Mellon) is listed under ticker "BNY" in EDGAR -> TICKER_CIK_OVERRIDES={"BK":
1390777}. (5) BNY reports pre-tax income only as Domestic/Foreign splits some
years -> BK EBIT (and interest_coverage / Altman EBIT term) is NaN for 2023.

[2026-06-20] BUG: Altman doc discrepancy — CLAUDE.md states "PNC 2023 verified:
Z = 1.847" but the documented fixture values (WC=18,RE=29,EBIT=6.1,MktCap=55,
Rev=22,TA=557,TL=504 in B) plugged into the documented formula compute to 0.253,
not 1.847. Real-data PNC 2023 Z=0.328 (WC term=0, banks have no working capital).
RECONCILE before finalizing risk_tools.score_altman_z and any test asserting
1.847. All 5 banks score < 1.81 (structural for leveraged banks — read relative
to peers, not absolute zones).

[2026-06-20] DECISION: notebooks/eda.ipynb built + executed end-to-end on REAL
API calls (no mocks). 5 banks (PNC,JPM,BK,WFC,BAC) x 2018-2023 = 30/30 rows
ingested. 5 charts -> reports/charts/eda_*.png (ratio_trends, zscore_distribution,
zscore_peer_2023, macro_overlay, correlation_heatmap). market_cap is yfinance
current-only, broadcast to all years (documented approximation).

[2026-06-20] BUG: CLAUDE.md stated PNC 2023 Z=1.847 as verified ground truth —
incorrect. Real EDGAR data gives Z=0.328, fixture values give Z=0.253. Both are
distress zone for banks but this is expected per the bank Z-Score caveat in
CLAUDE.md. Do not use 1.847 as a test assertion anywhere.

[2026-06-20] DECISION (Issue 2): dropped current_ratio from the ML feature set —
banks file unclassified balance sheets so AssetsCurrent/LiabilitiesCurrent (and
thus current_ratio & working_capital) are structurally NaN for all 5 banks.
feature_names.json created with 8 features (no current_ratio). working_capital
was never a feature (only an ingest field feeding altman_z). context_schema
IngestOutput marks current_assets/current_liabilities/working_capital as
Optional[float]; sec_tools FIELD_ALIASES carries an explanatory comment;
conftest pnc_2023_ingest fixture sets the three to None. Altman WC/TA term uses
WC=0 when working_capital is None.

[2026-06-20] DECISION (Issue 3): replaced current-only market cap with HISTORICAL
per-year market cap = dei:EntityCommonStockSharesOutstanding (SEC, selected by
fy not end-date — cover date falls in year+1) × year-end closing price
(yfinance, period="max"). sec_tools now loads full facts (us-gaap + dei), adds
_select_fiscal_year_value + _extract_shares_outstanding, returns
shares_outstanding. market_tools.fetch_market_data gained a period param
(default "1y", backward compatible). Notebook computes market_cap per row.
SIDE EFFECT: real-data PNC 2023 Altman Z moved 0.328 -> 0.284 (historical cap
$56.3B vs current $93B); CLAUDE.md Altman section updated. Still distress zone.

[2026-06-20] DECISION: accepted PNC 2023 Z = 0.284 (historical market cap) as the
real-data GROUND TRUTH, superseding 0.328 (which used yfinance current cap).
Fixture-values Z stays 0.253. Updated Altman Z-Score section + conftest docstring.
Ground-truth set is now {0.284 real, 0.253 fixture}; never assert 0.328 or 1.847.
Also moved the EDA notebook builder /tmp/build_eda.py -> src/utils/build_eda.py
(now tracked; output path derived from __file__; running it writes an UNEXECUTED
notebook that must then be executed to embed outputs/charts).

[2026-06-22] DECISION: analysis_tools.py implemented TDD (calc_ratios,
detect_anomalies, plot_chart; 7 tests). calc_ratios returns a Ratios dict with
current_ratio=None (dropped for banks, key kept per "null ok" contract).
interest_expense None is coerced to 0.0 then floored via max(_,0.001).
detect_anomalies/flag_risks emit dicts carrying a `code` key in addition to the
Anomaly/RiskFlag schema fields (extra keys are runtime-harmless for TypedDicts);
PNC fixture interest_coverage=6.1/4.2=1.452 (<1.5) so the high-severity path is
exercised by ground-truth values. plot_chart forces matplotlib's Agg backend
(headless) and saves to reports/charts/.

[2026-06-22] DECISION: risk_tools.py implemented TDD (score_altman_z,
run_ml_risk_model, flag_risks, compare_to_peers; 8 tests). RECONCILED the Altman
test discrepancy flagged on 06-20: a unit test of score_altman_z CANNOT assert
0.284 from fixture values — 0.284 is real-EDGAR ground truth (different line
items + historical cap), unreachable from any fixture-value combination
(reverse-engineering needs mktcap ~$81-114B). NOTE the conftest pnc_2023_ingest
fixture literally yields 0.214 because it sets working_capital=None (WC term=0);
0.253 only holds when WC=18B is plugged back in. Per user decision, the test
feeds documented fixture values WITH working_capital=18B and asserts Z=0.253,
zone=distress. run_ml_risk_model loads pkls via module-level _load_model /
_load_pipeline (pickle, not joblib — stdlib, no new requirement) so tests mock
those; no real .pkl exists yet (Week-2 train.py track still pending). It reads
feature order from feature_names.json, errors on any missing feature, and takes
predict_proba's last column as the high-risk probability (<0.34 low / <0.67
moderate / else high). compare_to_peers(z, peers) returns fraction of peers
at-or-below z in [0,1]. All 55 tests green.

[2026-06-22] RETRAIN: trained risk_classifier.pkl via src/models/train.py.
build_features.py (new, src/utils/) does live ingestion (same pipeline as the
EDA notebook) -> data/processed/features.csv (30 bank-year rows, 8 feature cols
in feature_names.json order). The notebook never persisted model_df, so this
script is now the canonical processed-data producer. After dropna: 20 rows (6
dropped for NaN altman_z where EBIT unavailable e.g. BNY; 5 for first-year
revenue_growth; overlap). StandardScaler+RandomForest(100,rs=42), stratified
80/20. Holdout accuracy = 1.000.
DECISION (labeling): absolute Altman zones collapse to a single 'high' class
(all banks Z~0.18-0.24, far below 1.81), which breaks 3-class training and the
predict_proba (1,3) test. Switched to PEER-RELATIVE TERTILES of altman_z
(lowest third->high, mid->moderate, top->low; 7/7/6 balance), per CLAUDE.md's
"read bank Z relative to peers" guidance.
CAVEAT (leakage): risk_label is derived from altman_z, and altman_z is also a
feature -> the model trivially recovers the tertile boundaries, so 1.000
accuracy is circular, not evidence of generalization. The >=60% gate passes but
is not meaningful under this scheme. Revisit before claiming predictive value
(drop altman_z from features, or label from an independent target).
CONVENTION: train.py saves the fitted pipeline with joblib; risk_tools
_load_model/_load_pipeline switched pickle->joblib to match (added joblib to
requirements.txt). feature_names.json stays JSON text (json.dump), NOT joblib,
because risk_tools._load_feature_names reads it via json.load.
OPEN ISSUE: train.py saves ONE combined Pipeline (scaler+clf) to
risk_classifier.pkl, but risk_tools.run_ml_risk_model still loads a SEPARATE
feature_pipeline.pkl (doesn't exist) then the model. Reconcile when wiring
RiskAgent: simplest is to have run_ml_risk_model load the one combined pipeline
and call predict_proba directly (drop the separate transform step).
---