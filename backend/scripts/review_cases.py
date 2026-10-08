"""Inspect candidate transcripts and accept explicit human review decisions.

Use --show ID to inspect the inputs, expected outputs and replay observations.
Use --case ID --reviewer NAME --decision approve|reject to record a human action.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from servemind.evaluation.curated_cases import CURATED_COMMERCE_CASES
from servemind.evaluation.human_review import REVIEW_PATH, case_digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show")
    parser.add_argument("--case")
    parser.add_argument("--reviewer")
    parser.add_argument("--decision", choices=("approve", "reject"))
    args = parser.parse_args()
    cases = {c["id"]: c for c in CURATED_COMMERCE_CASES}
    if args.show:
        from servemind.evaluation.evaluator import evaluate_curated_commerce
        print(json.dumps({"case": cases[args.show], "replay": evaluate_curated_commerce([cases[args.show]])}, ensure_ascii=False, indent=2))
    elif args.case and args.reviewer and args.decision:
        # Run only following an actual human review. Codex must never self-approve.
        reviews = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
        case = cases[args.case]
        reviews[args.case] = {"reviewer": args.reviewer, "reviewer_type": "human", "decision": args.decision,
                              "case_sha256": case_digest(case), "reviewed_at": datetime.now(timezone.utc).isoformat()}
        from pathlib import Path
        # User-invoked annotation command, not an agent code-edit workflow.
        REVIEW_PATH.write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps([{"id": c["id"], "turns": c["turns"], "review_status": c["annotation_status"]}
                          for c in CURATED_COMMERCE_CASES], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
