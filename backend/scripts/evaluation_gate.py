"""Local/deployment regression gate; paid Judge is opt-in only.

Usage: PYTHONPATH=src python3.12 scripts/evaluation_gate.py [--live-judge]
"""
from __future__ import annotations

import argparse
import json

from servemind.agents.agent_orchestrator import ServeMindAgent
from servemind.evaluation.evaluator import evaluate_all


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-judge", action="store_true",
                        help="Use configured DeepSeek API and incur provider usage")
    args = parser.parse_args()
    report = evaluate_all(ServeMindAgent(), live_judge=args.live_judge)
    compact = {
        "overall_score_legacy_formula": report["overall_score"],
        "levels": report["levels"],
        "task_graph": {key: report["task_graph"][key]
                       for key in ("total", "passed", "accuracy", "task_completion_rate")},
        "legacy_multi_intent": {key: report["multi_intent"][key]
                                for key in ("total", "passed", "accuracy",
                                            "semantic_passed", "semantic_accuracy")},
        "judge_usage": report["llm_judge"]["usage"],
        "natural_dialogue": {key: report["curated_commerce"][key] for key in ("total", "passed", "accuracy", "human_review_gate")},
        "regression_gate": report["regression_gate"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0 if report["regression_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
