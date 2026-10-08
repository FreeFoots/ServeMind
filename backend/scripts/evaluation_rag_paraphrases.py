"""Read-only natural-query retrieval diagnostic; candidate labels, not human gold.

Uses the configured Qwen/pgvector/reranker stack. No answer generation, query
rewriting, or LLM Judge is involved, so this is not an end-to-end RAG score.
Existing knowledge and approval records are never modified.
"""
import argparse
import json
import time
from pathlib import Path

from servemind.config.settings import PROJECT_ROOT
from servemind.core.knowledge_base import KnowledgeBase


CASES = [
    ("buyer_and_merchant", "other", ("买家和店家分别能咨询什么事情？", "店主想查询仓库和履约的总体情况，可以问客服吗？")),
    ("cancellation_review", "cancel_policy", ("还没收到东西，现在能撤销订单吗？", "我提出取消了，是不是就代表已经取消成功？")),
    ("commerce_boundaries", "other", ("系统里能找到库存记录，是不是就证明现在有现货？", "只有一条订单记录，能确定我已经付款了吗？")),
    ("delivery_exception", "complaint", ("物流显示送达但我根本没拿到包裹，怎么办？", "包裹好几天没动静了，可以直接认定丢件并赔偿吗？")),
    ("merchant_handoff", "human_handoff", ("你处理不了时，怎么把我的情况交给这家店的客服？", "我想和店家本人谈，AI可以代替他批准退款吗？")),
    ("privacy_and_identity", "human_handoff", ("店家现在能看见我和AI之前的私聊吗？", "我能不能看到另一个买家的订单和聊天记录？")),
    ("refund_review", "refund_policy", ("申请退钱之前，需要谁核实金额和资格？", "没有退款流水，能不能告诉我钱已经到账了？")),
    ("invoice_support", "invoice_query", ("我想开公司的票，要先提供哪些信息？", "个人抬头和单位抬头，客服会怎样帮我联系店家？")),
    ("return_preparation", "refund_policy", ("退货前我需要准备什么资料，东西寄去哪里？", "店家还没给退货地址，我可以直接把东西寄回去吗？")),
    ("delivery_schedule", "delivery_status", ("页面没有预计送达日期，能保证哪天收到吗？", "说已经发货了，是否就表示已经送到我这里？")),
    ("stock_and_presale", "inventory_query", ("预售和暂时没货有什么不同，发货时间找谁问？", "页面写在售，是不是保证下单时一定买得到？")),
    ("product_attributes", "sku_query", ("商品没写材质和尺寸，可以凭类别帮我判断吗？", "想知道耳机能不能兼容我的手机，页面没注明怎么办？")),
    ("price_and_discounts", "price_breakdown", ("页面标价和我真正付款的金额为什么不一样？", "没有成交记录，能推算优惠券叠加后少付多少钱吗？")),
    ("address_change", "human_handoff", ("能替我换收货地点吗，完整住址该发在聊天里吗？", "包裹已经发出，谁能确认还可不可以改收货人？")),
    ("handoff_followup", "human_handoff", ("店家回复了，但问题没解决，我还能继续追问吗？", "已经交给人工处理了，收到回复就算结案了吗？")),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "backend/runtime/rag-paraphrases.json")
    args = parser.parse_args()
    knowledge = KnowledgeBase()
    details = []
    for expected, intent, queries in CASES:
        for query in queries:
            started = time.monotonic()
            result = knowledge.search_for_intent(query, intent, top_k=3)
            titles = [item["title"] for item in result["items"]]
            row = {"query": query, "intent": intent, "expected_document": expected,
                   "titles": titles, "hit_at_1": bool(titles) and titles[0] == expected,
                   "hit_at_3": expected in titles,
                   "required_document_precision_at_3": sum(t == expected for t in titles) / len(titles) if titles else 0,
                   "backend": result.get("backend"), "cache_hit": result.get("cache_hit"),
                   "latency_ms": round((time.monotonic() - started) * 1000, 2),
                   "reranked": any(item.get("retrieval", {}).get("reranked") for item in result["items"])}
            details.append(row)
            print(json.dumps({"completed": len(details), "query": query, "hit_at_3": row["hit_at_3"]}, ensure_ascii=False), flush=True)
    report = {"protocol": "natural-policy-retrieval-diagnostic-v1", "human_gold": False,
              "scope": "retrieval_only_without_llm_rewrite_or_answer_generation",
              "label_note": "one required document per query; other relevant documents are not labeled",
              "knowledge_version": knowledge.corpus_version, "loaded_documents": len(knowledge.documents),
              "vector_index": knowledge.vector.summary() if knowledge.vector else None,
              "total": len(details),
              "hit_at_1": sum(r["hit_at_1"] for r in details) / len(details),
              "hit_at_3": sum(r["hit_at_3"] for r in details) / len(details),
              "required_document_precision_at_3": sum(r["required_document_precision_at_3"] for r in details) / len(details),
              "fallback_count": sum("fallback" in str(r["backend"]) for r in details),
              "cases": details, "release_gate_passed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.chmod(0o600)
    print(json.dumps({k: report[k] for k in ("total", "hit_at_1", "hit_at_3", "required_document_precision_at_3", "fallback_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
