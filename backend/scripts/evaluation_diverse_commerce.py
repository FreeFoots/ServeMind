"""Real-provider regression across existing synthetic products and merchants.

Preserves the original candidate expectations. Literal checks may reject valid
paraphrases; report them separately from Judge instead of changing gold labels.
No production product/message records or human approvals are changed.
"""
import argparse
import json
from pathlib import Path

from servemind.config.settings import PROJECT_ROOT
from servemind.evaluation.expanded_cases import EXPANDED_COMMERCE_CASES
from evaluation_model_regression import build_report


TOPICS = (
    ["price_breakdown"],
    ["inventory_query", "price_breakdown"],
    ["inventory_query"],
    ["inventory_query", "price_breakdown"],
    ["invoice_query"],
    ["complaint", "inventory_query"],
    ["human_handoff", "price_breakdown"],
    ["refund_policy"],
    ["delivery_status", "price_breakdown"],
    ["sku_query", "price_breakdown"],
    ["delivery_exception", "refund_policy"],
    ["price_breakdown", "delivery_status"],
)


def candidates():
    selected = []
    for product_index in range(20):
        # One two-intent stock/price question and one rotating risk/feature case.
        variant = (product_index % 11) + 2
        if variant == 12:
            variant = 0
        for v in (1, variant):
            case = EXPANDED_COMMERCE_CASES[product_index * 12 + v]
            selected.append({"id": case["id"], "author_type": "agent_candidate",
                             "annotation_status": "awaiting_human_review",
                             "product": case["product"], "purchase": case.get("purchase"),
                             "turns": [{"message": case["message"], "topics": TOPICS[v],
                                        "handoff": case["handoff"], "contains": case["required"]}]})
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Explicit paid generation and Judge")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "backend/runtime/diverse-commerce.json")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live required")
    cases = candidates()
    report = build_report(cases, live=True)
    report["dataset_coverage"] = {"products": len({c["product"]["id"] for c in cases}),
                                  "merchants": len({c["product"]["merchant_id"] for c in cases}),
                                  "categories": sorted({c["product"]["category"] for c in cases}),
                                  "stock_states": sorted({c["product"]["catalog_status"] for c in cases}),
                                  "purchase_states": sorted({c["purchase"]["fulfillment_status"] for c in cases if c["purchase"]}),
                                  "identity_scope": "synthetic replay, not authenticated multi-buyer production traffic"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"], "release_gate": report["release_gate"],
                      "dataset_coverage": report["dataset_coverage"]}, ensure_ascii=False), flush=True)
    return 0 if report["release_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
