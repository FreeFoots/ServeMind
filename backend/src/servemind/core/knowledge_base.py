from __future__ import annotations

from pathlib import Path
import os
import json
import hashlib
import threading
from datetime import date

from servemind.config.settings import KNOWLEDGE_ROOT
from servemind.config.policy_registry import POLICY_VERSION
from servemind.core.vector_knowledge import VectorKnowledgeBase
from servemind.memory.redis_query_cache import RedisQueryCache

POLICY_INTENTS = {'refund_policy','cancel_policy','complaint','human_handoff','invoice_query','other',
                  'inventory_query','sku_query','price_breakdown','delivery_status','delivery_exception','order_query'}


class KnowledgeBase:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or KNOWLEDGE_ROOT
        self.documents: list[dict[str, str]] = []
        self.vector = (VectorKnowledgeBase() if os.getenv("SERVEMIND_RAG_BACKEND") == "pgvector"
                       else None)
        self.cache = RedisQueryCache() if self.vector else None
        self.last_backend = "keyword"
        self.last_cache_hit = False
        self._lock = threading.RLock()
        self.reload()

    def reload(self) -> None:
        self.documents = []
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        versions = manifest.get("documents", manifest)
        today = date.today()
        if self.root.exists():
            for path in sorted(self.root.glob("*.md")):
                metadata = versions.get(path.name, versions.get(path.stem, {}))
                start = date.fromisoformat(metadata["valid_from"]) if metadata.get("valid_from") else None
                end = date.fromisoformat(metadata["valid_until"]) if metadata.get("valid_until") else None
                if (start and today < start) or (end and today > end):
                    continue
                content = path.read_text(encoding="utf-8")
                display_title = next((line[2:].strip() for line in content.splitlines()
                                      if line.startswith("# ")), path.stem)
                default_intents = {"refund_review": ["refund_policy"], "cancellation_review": ["cancel_policy"],
                                   "delivery_exception": ["complaint", "human_handoff"],
                                   "merchant_handoff": ["complaint", "human_handoff"],
                                   "privacy_and_identity": ["human_handoff", "complaint"],
                                   "commerce_boundaries": ["other"], "buyer_and_merchant": ["other"]}
                self.documents.append({"title": path.stem, "display_title": display_title,
                                       "content": content, "document_version": metadata.get("version", "1.0.0"),
                                       "intents": metadata.get("intents", default_intents.get(path.stem, ["other"])),
                                       "customer_guidance": content.split("## 客服说明", 1)[1].strip() if "## 客服说明" in content else "",
                                       "valid_from": metadata.get("valid_from"),
                                       "valid_until": metadata.get("valid_until")})

        self.corpus_version = hashlib.sha256(json.dumps(self.documents, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]

    def search(self, query: str, top_k: int = 3, *, document_ids: list[str] | None = None) -> list[dict[str, object]]:
        self.last_cache_hit = False
        if self.vector is not None:
            cache_key_backend = "hybrid_qwen3_v4:" + self.corpus_version + ":" + str(document_ids) + ":" + date.today().isoformat()
            cached = self.cache.get(query, top_k=top_k, version=POLICY_VERSION,
                                    backend=cache_key_backend) if self.cache else None
            if cached is not None:
                self.last_backend = "redis_cache_hybrid_qwen3"
                self.last_cache_hit = True
                return cached
            try:
                items = self.vector.search(query, top_k=top_k, document_ids=document_ids)
                metadata = {item["title"]: item for item in self.documents}
                for item in items:
                    item["customer_guidance"] = metadata.get(item["title"], {}).get("customer_guidance", "")
                    item["policy_intents"] = metadata.get(item["title"], {}).get("intents", [])
                self.last_backend = "hybrid_qwen3"
                if self.cache:
                    self.cache.put(query, items, top_k=top_k, version=POLICY_VERSION,
                                   backend=cache_key_backend)
                return items
            except Exception:
                # The knowledge store must not take down a buyer conversation.
                # The fallback is explicitly reported, never presented as vector RAG.
                self.last_backend = "keyword_fallback"
        else:
            self.last_backend = "keyword"
        import re
        text = (query or "").lower()
        words = {word for word in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9_]{2,}", text) if len(word) > 1}
        words.update(phrase[i:i+2] for phrase in re.findall(r'[\u4e00-\u9fff]{2,}',text)
                     for i in range(len(phrase)-1))
        expansions = {
            "退款": {"退款", "退货", "支付", "人工"},
            "取消": {"取消", "订单", "人工"},
            "投诉": {"投诉", "边界", "人工"},
            "人工": {"人工", "升级", "确认"},
            "不知道": {"无法", "确认", "订单", "商品"},
        }
        for marker, terms in expansions.items():
            if marker in text:
                words.update(terms)
        scored = []
        for document in self.documents:
            if document_ids is not None and document["title"] not in document_ids:
                continue
            content = document["content"].lower()
            score = sum(1 for word in words if word in content)
            if score:
                scored.append({"title": document["title"], "display_title": document["display_title"],
                               "content": document["content"][:1000],
                               "score": round(score / max(len(words), 1), 3),
                               "source": f"knowledge/{document['title']}.md",
                               "document_version": document["document_version"],
                               "customer_guidance": document["customer_guidance"], "policy_intents": document["intents"],
                               "valid_from": document["valid_from"],
                               "valid_until": document["valid_until"]})
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

    def search_for_intent(self, query: str, intent: str, top_k: int = 3,
                          queries: list[str] | None = None) -> dict[str, object]:
        """意图驱动 RAG：仅在政策/售后意图下检索知识库，并保留路由信息。"""
        allowed = POLICY_INTENTS
        if intent not in allowed:
            return {"enabled": False, "intent": intent, "items": [], "reason": "business_fact_intent_uses_deterministic_tools"}
        document_ids = [doc["title"] for doc in self.documents if intent in doc["intents"]]
        with self._lock:
            variants = list(dict.fromkeys([query, *(queries or [])]))[:3]
            fused, scores, backends, hits = {}, {}, set(), []
            for variant in variants:
                items = self.search(variant, top_k=max(top_k, 5), document_ids=document_ids)
                backends.add(self.last_backend)
                hits.append(self.last_cache_hit)
                for rank, item in enumerate(items, 1):
                    key = item.get('chunk_id') or item['title']
                    fused[key] = item
                    scores[key] = scores.get(key, 0) + 1/(60+rank)
            items = [{**fused[k], 'query_fusion_score':round(scores[k],6)}
                     for k in sorted(scores, key=scores.get, reverse=True)[:top_k]]
            return {"enabled": True, "intent": intent, "items": items, "reason": "policy_or_boundary_intent",
                    "backend": '+'.join(sorted(backends)), "cache_hit": all(hits),
                    "query_count":len(variants), "fusion":"reciprocal_rank_60"}

    def summary(self) -> dict[str, object]:
        return {"documents": len(self.documents), "titles": [item["title"] for item in self.documents],
                "backend": "hybrid_qwen3" if self.vector else "keyword"}
