"""Save reproducible component results plus optional real commercial generation/Judge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from servemind.agents.agent_orchestrator import ServeMindAgent
from servemind.config.settings import PROJECT_ROOT
from servemind.evaluation.evaluator import evaluate_all, evaluate_curated_commerce


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-curated", action="store_true", help="Use configured DeepSeek for natural conversations and six-dimensional Judge")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "backend" / "runtime" / "evaluation-latest.json")
    args = parser.parse_args()
    report = evaluate_all(ServeMindAgent(), live_judge=args.live_curated)
    if args.live_curated:
        print("Component regression completed; running real DeepSeek natural-dialogue generation and Judge", flush=True)
        report["live_commerce"] = evaluate_curated_commerce(live_generation=True, live_judge=True)
        live = report["live_commerce"]
        checks = report["regression_gate"]["checks"]
        checks["live_natural_dialogue"] = live["accuracy"] >= 0.9
        checks["live_judge_quality"] = live["live_judge"]["mean"] is not None and live["live_judge"]["mean"] >= 0.85
        checks['live_judge_coverage']=live['live_judge']['success_rate']==1
        checks["live_provider_usage"] = live["runtime_metrics"]["provider_usage"]["input_tokens"] > 0
        # Model execution must pass independently of deterministic fallback.
        # Template correctness cannot stand in for successful specialist analysis.
        checks["live_model_agents_active"] = live["runtime_metrics"].get("model_agent_requests", 0) > 0
        checks["live_specialist_model_success"] = live["runtime_metrics"].get("specialist_model_success_rate", 0) >= .9
        checks["live_intent_recognition"] = live["runtime_metrics"]["intent_recognition_fallback_count"] == 0
        checks["live_tool_governance"] = live["runtime_metrics"].get("unauthorized_block_count", 0) == 0
        report["regression_gate"]["passed"] = all(checks.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "components": report["components"], "gate": report["regression_gate"],
                      "natural": {k: report["curated_commerce"][k] for k in ("total", "passed", "accuracy", "human_review_gate")},
                      "live_natural": {k: report["live_commerce"][k] for k in ("total", "passed", "accuracy", "live_judge")} if args.live_curated else None}, ensure_ascii=False, indent=2))
    return 0 if report["regression_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
