"""run.py — FinSight CLI entry point.

Usage:
    python run.py "Analyze PNC 2023"
    python run.py "Analyze PNC 2023" --track-cost
"""

import sys

from src.agents.orchestrator import OrchestratorAgent


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: python run.py 'Analyze <TICKER> <YEAR>' [--track-cost]")
        sys.exit(1)

    track_cost = "--track-cost" in args
    query_args = [a for a in args if not a.startswith("--")]
    if not query_args:
        print("Error: no query provided.")
        sys.exit(1)

    query = " ".join(query_args)

    agent = OrchestratorAgent()
    result = agent.run(query)

    if result.get("status") == "failed":
        print(f"[FinSight] FAILED: {result.get('error', 'unknown error')}")
        print(f"  run_id : {result.get('run_id', '')}")
        sys.exit(1)

    risk = result.get("risk", {})
    print("[FinSight] Pipeline complete")
    print(f"  run_id      : {result['run_id']}")
    print(f"  report_path : {result['report_path']}")
    print(f"  z_score     : {risk.get('z_score', 'n/a')} ({risk.get('z_zone', '')})")
    print(f"  ml_risk     : {risk.get('ml_risk_label', 'n/a')} "
          f"(prob={risk.get('ml_risk_prob', 'n/a')})")
    print(f"  peer_pct    : {risk.get('peer_percentile', 'n/a')}")

    if track_cost:
        # Token tracking is best-effort — agents don't yet surface usage objects
        print("\n[--track-cost] Token tracking not yet wired into agents.")
        print("  Set up usage accumulation in OrchestratorAgent to enable this.")

    sys.exit(0)


if __name__ == "__main__":
    main()
