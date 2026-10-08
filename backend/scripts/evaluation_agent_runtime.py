"""Small explicit paid smoke/Judge, not a thousand-case release evaluation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from servemind.agents.model_runtime import parse_json, public_packet
from servemind.config.settings import PROJECT_ROOT
from servemind.evaluation.commerce_cases import PRODUCT
from servemind.service.commerce_support import CommerceSupport

CASES = [
    ("agent-live-stock-price", "这款还能买到吗？多少钱？", {"inventory_query", "price_breakdown"}, False),
    ("agent-live-private-policy", "先告诉我商品多少钱，再说明怎么退货，先了解规则，别通知商家。", {"price_breakdown", "refund_policy"}, False),
    ("agent-live-handoff", "我要退款，请商家在这里处理。", {"refund_policy"}, True),
]


class NoMemory:
    def get(self, key):
        return {}
    def put(self, *args):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Explicitly call configured DeepSeek and real Judge")
    parser.add_argument("--limit", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "backend/runtime/agent-runtime-smoke.json")
    args = parser.parse_args()
    if not args.live:
        parser.error("pass --live to authorize this paid smoke; fixtures run through pytest instead")
    support = CommerceSupport(working_memory=NoMemory())
    if not support.model_runtime:
        raise RuntimeError("configured_model_agent_runtime_required")
    records = []
    for identifier, question, expected, handoff in CASES[:args.limit]:
        answer, meta = support.reply(question, PRODUCT)
        quality, judge_usage, judge_error = None, None, None
        try:
            completion = support.llm.complete(system=(
                "你是独立电商客服评审。根据提供的证据、诉求、预期交接判断实际回答；不采信 Agent 自评分。"
                "只输出 JSON，六项分数各为 0 到 1：relevance,accuracy,completeness,helpfulness,grounding,handoff。"
                "不能因提到证据编号就给满分。不得把未办理退款当成功，不应将仅了解规则的私聊交给商家。"),
                user=json.dumps(public_packet({"question": question, "answer": answer,
                    "evidence": meta["evidence"], "expected_topics": sorted(expected),
                    "expected_handoff": handoff, "actual_handoff": meta["needs_merchant"]}), ensure_ascii=False),
                max_tokens=300)
            # Preserve usage even if parsing or the Judge contract fails.
            judge_usage = {"model": completion.model, "prompt_tokens": completion.prompt_tokens,
                           "completion_tokens": completion.completion_tokens}
            scores = parse_json(completion.text)
            keys = ("relevance", "accuracy", "completeness", "helpfulness", "grounding", "handoff")
            if not isinstance(scores, dict) or not all(type(scores.get(k)) in (int, float)
                and math.isfinite(scores[k]) and 0 <= scores[k] <= 1 for k in keys):
                raise ValueError("judge_contract_failed")
            quality = {"scores": {k: scores[k] for k in keys}, "mean": sum(scores[k] for k in keys)/6}
        except Exception as exc:
            judge_error = type(exc).__name__
        contributions = meta.get("agent_runtime", {}).get("contributions", [])
        checks = {"topic_coverage": expected.issubset(meta["topics"]),
            "handoff_correct": meta["needs_merchant"] == handoff,
            "grounded": meta["grounding"]["passed"],
            "intent_recognition_used": meta["intent_three_way"].get("mode") == "model_intent_recognition",
            "parallel_role_routing": meta["task_graph"].get("mode") == "model_role_parallel" and all(not n["depends_on"] for n in meta["task_graph"]["nodes"]),
            "specialists_used": bool(contributions) and all(c["status"] == "model_analyzed" for c in contributions),
            "composer_accepted": not meta["model_use"].get("output_rejected") and any(
                c["stage"] == "response_composer" for c in meta.get("agent_runtime", {}).get("calls", [])),
            "tool_governance": all(t["status"] == "ok" for t in meta["tools_used"]),
            "judge_quality": quality is not None and quality["mean"] >= .85}
        records.append({"id": identifier, "question": question, "answer": answer, "checks": checks,
            "passed": all(checks.values()), "quality": quality, "judge_usage": judge_usage,
            "judge_error": judge_error, "metadata": meta})
        print(json.dumps({"id": identifier, "passed": records[-1]["passed"], "checks": checks}, ensure_ascii=False), flush=True)
    report = {"kind": "real_provider_synthetic_smoke_not_release", "total": len(records),
        "passed": sum(r["passed"] for r in records), "cases": records,
        "runtime_metrics": support.monitor.summary(), "human_reviewed": False,
        "release_gate_passed": False, "release_gate_reason": "small_unreviewed_sample_and_full_regression_required"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": report["passed"], "total": report["total"]}, ensure_ascii=False))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
