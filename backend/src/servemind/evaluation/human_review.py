"""Human labels bind to immutable case contents; agent candidates are never approvals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from servemind.config.settings import PROJECT_ROOT
from servemind.evaluation.curated_cases import CURATED_COMMERCE_CASES

REVIEW_PATH = PROJECT_ROOT / "backend" / "evaluation" / "human_reviews.json"


def case_digest(case: dict) -> str:
    return hashlib.sha256(json.dumps(case, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def human_review_gate(results: list[dict], *, reviews: dict | None = None, minimum: int = 20) -> dict:
    if reviews is None:
        reviews = json.loads(REVIEW_PATH.read_text(encoding="utf-8")) if REVIEW_PATH.exists() else {}
    approved, pending, rejected = [], [], []
    for case in CURATED_COMMERCE_CASES:
        review = reviews.get(case["id"], {})
        if (review.get("case_sha256") != case_digest(case) or not review.get("reviewer")
                or review.get("reviewer_type") != "human"):
            pending.append(case["id"])
        elif review.get("decision") == "approve":
            approved.append(case["id"])
        else:
            rejected.append(case["id"])
    observations = {row["id"]: row for row in results}
    passed = sum(bool(observations.get(cid, {}).get("passed")) for cid in approved)
    critical_failures = [c["id"] for c in CURATED_COMMERCE_CASES
                         if c["risk"] == "critical" and c["id"] in approved and not observations.get(c["id"], {}).get("passed")]
    accuracy = round(passed / len(approved), 4) if approved else None
    return {"passed": len(approved) >= minimum and accuracy >= 0.9 and not critical_failures,
            "minimum_human_approved": minimum, "approved": len(approved), "pending": len(pending),
            "rejected": len(rejected), "approved_accuracy": accuracy, "critical_failures": critical_failures,
            "pending_ids": pending, "reason": "awaiting_human_annotations" if len(approved) < minimum else "reviewed_case_regression"}
