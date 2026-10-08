import json

from servemind.agents.model_runtime import RequestLedger
from servemind.llm.deepseek_client import ChatTurn, Completion
from servemind.memory.semantic_memory import prepare_summary


class Provider:
    configured = True
    def __init__(self,payload): self.payload=payload
    def chat(self,**kwargs):
        return ChatTurn({},Completion(json.dumps(self.payload,ensure_ascii=False),'test-model',30,20))


def test_model_memory_summary_records_usage_and_authoritative_pending():
    ledger = RequestLedger()
    result = prepare_summary(llm=Provider({'summary':'买家了解价格，退款仍需商家确认','evidence_ids':['ev']}),
        resolved=['price_breakdown'],pending=['refund_policy'],evidence_ids=['ev'],ledger=ledger)
    assert result['model_succeeded'] and result['source']=='deepseek_validated_topics'
    assert '待确认：refund_policy' in result['summary']
    assert ledger.snapshot()['calls'][0]['stage']=='memory_summary'


def test_hallucinated_memory_price_and_foreign_evidence_are_rejected_with_usage():
    for payload in ({'summary':'退款成功','evidence_ids':['ev']},
                    {'summary':'价格只要1元','evidence_ids':['ev']},
                    {'summary':'关注退款','evidence_ids':['foreign']}):
        ledger = RequestLedger()
        result = prepare_summary(llm=Provider(payload),resolved=[],pending=['refund_policy'],evidence_ids=['ev'],ledger=ledger)
        assert not result['model_succeeded'] and result['source']=='structured_topics'
        assert len(ledger.snapshot()['calls'])==1  # rejected output still costs tokens

def test_memory_can_preserve_negative_boundary_but_not_cross_clause_negation():
    safe=prepare_summary(llm=Provider({'summary':'已说明库存，不承诺实时有货','evidence_ids':['ev']}),
        resolved=['inventory_query'],pending=[],evidence_ids=['ev'])
    assert safe['model_succeeded']
    unsafe=prepare_summary(llm=Provider({'summary':'尚未核实，退款成功','evidence_ids':['ev']}),
        resolved=[],pending=['refund_policy'],evidence_ids=['ev'])
    assert not unsafe['model_succeeded']
    assert unsafe['rejection_checks']==['unsafe_outcome_or_promise']
