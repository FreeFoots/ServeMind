import json
from types import SimpleNamespace

import pytest

from test_model_agent_runtime import ScriptedModel
from servemind.agents.model_runtime import RequestLedger
from servemind.core.commerce_response import validate_sections
from servemind.core.knowledge_base import KnowledgeBase
from servemind.evaluation.commerce_cases import PRODUCT
from servemind.evaluation.curated_cases import CURATED_COMMERCE_CASES
from servemind.evaluation.evaluator import evaluate_curated_commerce
from servemind.evaluation.metrics import classification_metrics, baseline_gate, wilson_lower
from servemind.memory.semantic_memory import prepare_summary
from servemind.monitor.operations import PerformanceMonitor, AnomalyDetector
from servemind.service.commerce_support import CommerceSupport

def test_semantic_entailment_rejects_unsupported_prose_and_counts_verifier_usage():
    from servemind.llm.deepseek_client import Completion
    class Model(ScriptedModel):
        def verify_claims(self,packet,**kwargs):
            return Completion(json.dumps({'supported':False,'unsupported_sections':['price_breakdown']}),'fixture',30,10)
    answer,meta=CommerceSupport(llm_client=Model(['price_breakdown'])).reply('多少钱',PRODUCT)
    assert '99.00' in answer and meta['model_use']['output_rejected']
    assert meta['model_use']['prompt_tokens']==110
    assert meta['grounding']['semantic_check']['supported'] is False

def test_third_tool_round_can_be_followed_by_tool_disabled_final_analysis():
    from servemind.llm.deepseek_client import Completion, ChatTurn
    class Model(ScriptedModel):
        def chat(self,*,messages,tools=None,**kwargs):
            system=messages[0]['content']
            if '公共政策检索查询改写器' in system:
                return ChatTurn({},Completion('[]','fixture',5,5))
            if '专业客服 Agent' not in system:
                return super().chat(messages=messages,tools=tools,**kwargs)
            previous=[m for m in messages if m['role']=='tool']
            if len(previous)<3:
                index=len(previous)
                name=['get_current_product','search_knowledge','get_current_purchase'][index]
                product_id=json.loads(messages[1]['content'])['product_id']
                params={'product_id':product_id} if index!=1 else {'query':'价格优惠','intent':'price_breakdown'}
                packet={'role':'assistant','content':None,'tool_calls':[{'id':f'call-{index}',
                        'function':{'name':name,'arguments':json.dumps(params)}}]}
            else:
                assert kwargs['tool_choice']=='none'
                refs=[e['evidence_id'] for m in previous for e in json.loads(m['content'])['evidence']]
                packet={'role':'assistant','content':json.dumps({'analysis':'已核实价格与规则','evidence_ids':refs})}
            return ChatTurn(packet,Completion(packet.get('content') or '','fixture',20,10))
    _,meta=CommerceSupport(llm_client=Model(['price_breakdown'])).reply('多少钱',PRODUCT)
    assert meta['task_graph']['model_success_rate']==1
    assert meta['agent_contributions'][0]['status']=='model_analyzed'

def test_confirmed_connection_error_retried_once_not_fallback():
    import httpx
    class Model(ScriptedModel):
        failures=0
        def chat(self,*,messages,**kwargs):
            if '专业客服 Agent' in messages[0]['content'] and not self.failures:
                self.failures+=1
                raise httpx.ConnectError('fixture')
            return super().chat(messages=messages,**kwargs)
    _,meta=CommerceSupport(llm_client=Model(['price_breakdown'])).reply('多少钱',PRODUCT)
    assert meta['task_graph']['model_success_rate']==1
    assert 'task-1:transport_retry' in meta['agent_runtime']['errors']

def test_dependency_policy_is_projected_to_current_topic_scope():
    from servemind.agents.task_graph import TaskNode
    from servemind.mcp.commerce_tools import ROLE_SCOPES
    model=ScriptedModel(['price_breakdown'])
    support=CommerceSupport(llm_client=model)
    node=TaskNode('task-1','price_breakdown','billing','get_current_product',{'product_id':PRODUCT['id']})
    context={'product':PRODUCT,'permissions':{'catalog_read','policy_public','purchase_read'},
             'allowed_tools':set(ROLE_SCOPES['billing']), 'dependency_evidence':[
                 {'evidence_id':'stock-policy','kind':'knowledge','policy_intents':['inventory_query'],'field':'policy','value':'库存政策'}]}
    result=support.model_runtime._run_node(node,'多少钱',context,RequestLedger())
    assert result.success and not result.degraded
    packet=json.loads(model.requests[0][1]['content'])
    assert not packet['dependency_evidence']

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('SERVEMIND_RAG_BACKEND','keyword')
    monkeypatch.setenv('SERVEMIND_SEMANTIC_INTENT','false')
    monkeypatch.setenv('SERVEMIND_AGENT_MODE','model')

@pytest.mark.parametrize('semantic',[False,True])
def test_expired_summary_is_noncritical_and_skips_model_and_embedding(semantic,monkeypatch):
    monkeypatch.setattr('servemind.memory.semantic_memory.encode_document',lambda _:pytest.fail('embedding called'))
    result=prepare_summary(llm=ScriptedModel(),resolved=['price_breakdown'],pending=['refund_policy'],
                           evidence_ids=[],ledger=RequestLedger(-1),semantic=semantic)
    assert not result['model_attempted'] and not result['embedding_ready']
    assert result['fallback_reason']=='request_budget_exhausted'

def test_actual_model_curated_path_has_completion_contract(monkeypatch):
    monkeypatch.setattr('servemind.service.commerce_support.DeepSeekClient',lambda:ScriptedModel(['inventory_query','price_breakdown']))
    report=evaluate_curated_commerce([CURATED_COMMERCE_CASES[0]],live_generation=True)
    graph=report['cases'][0]['turns'][0]['task_graph']
    assert graph['completion_rate']==1 and graph['model_success_rate']==1
    assert report['execution_scope']=='commercial_model_domain_routing_parallel_roles_scoped_replay_memory'

def test_recent_merchant_text_is_attributed_in_model_context():
    model=ScriptedModel(['price_breakdown'])
    CommerceSupport(llm_client=model).reply('按他刚才说的来',PRODUCT,
        recent_messages=[{'sender_type':'merchant','content':'蓝色可以，明天发','metadata':{}}])
    context=json.loads(model.requests[0][1]['content'])['context']['recent_messages']
    assert context[0]['text']=='蓝色可以，明天发'
    assert context[0]['authority']=='merchant_statement_unverified'

def test_explicit_second_intent_not_erased_by_model():
    _,meta=CommerceSupport(llm_client=ScriptedModel(['price_breakdown'])).reply('多少钱，有货吗',PRODUCT)
    assert set(meta['topics'])=={'price_breakdown','inventory_query'}

def test_atomic_claim_value_cannot_be_changed_or_bound_to_unrelated_field():
    ev=[{'evidence_id':'ev','kind':'product','field':'display_price','value':'99'}]
    section={'intent':'price_breakdown','status':'answered','evidence_ids':['ev'],
             'claims':[{'field':'display_price','value':'1','kind':'product','evidence_id':'ev'}]}
    assert not validate_sections([section],ev)['passed']
    section['claims']=[]
    section['intent']='inventory_query'
    assert not validate_sections([section],ev)['passed']

def test_multiquery_policy_fusion_keeps_intent_document_scope():
    kb=KnowledgeBase()
    result=kb.search_for_intent('库存预售','inventory_query',queries=['缺货有货','预售库存'])
    assert result['query_count']==3
    assert result['items'] and all(i['title']=='stock_and_presale' for i in result['items'])
    assert kb.search_for_intent('价格优惠','price_breakdown')['enabled']

def test_classification_macro_metrics_and_confusion_not_just_accuracy():
    metrics=classification_metrics(['a','a','b'],['a','a','a'])
    assert metrics['accuracy']==2/3
    assert metrics['per_class']['b']['recall']==0
    assert metrics['macro_f1']<metrics['accuracy']
    assert metrics['confusion_matrix']['b']['a']==1

def test_real_supervisor_evaluator_does_not_use_rule_predictions():
    from servemind.evaluation.evaluator import evaluate_model_intents
    support=CommerceSupport(llm_client=ScriptedModel(['inventory_query']))
    result=evaluate_model_intents(support,[{'id':'sample','message':'多少钱','expected':'price_breakdown'}])
    assert result['cases'][0]['predicted']=='inventory_query'
    assert result['classification']['accuracy']==0
    assert result['cases'][0]['provider_calls']

def test_policy_binding_cannot_reuse_different_intent_rule():
    ev=[{'evidence_id':'policy','kind':'knowledge','field':'policy','value':'库存规则','policy_intents':['inventory_query']}]
    section={'intent':'refund_policy','status':'answered','evidence_ids':['policy']}
    assert not validate_sections([section],ev)['passed']

def test_baseline_scope_hash_and_metric_degradation_are_enforced():
    baseline={'protocol':'v2','dataset_sha256':'x','model':'m','execution_scope':'live','metrics':{'f1':.9}}
    assert baseline_gate(baseline,baseline)['passed']
    assert not baseline_gate({**baseline,'execution_scope':'proxy'},baseline)['passed']
    assert not baseline_gate({**baseline,'metrics':{'f1':.7}},baseline)['passed']
    assert wilson_lower(3,3)<.75

def test_monitor_exports_actual_observations_and_updates_router(tmp_path):
    support=CommerceSupport(llm_client=ScriptedModel(['price_breakdown']))
    monitor=PerformanceMonitor(support,log_path=tmp_path/'alerts.jsonl',webhook='')
    support.reply('多少钱',PRODUCT)
    payload=monitor.render().decode()
    assert 'servemind_request_latency_seconds_count 1.0' in payload
    assert 'servemind_model_tokens_total{direction="input"} 80.0' in payload
    support.model_runtime._health['billing:primary']={'total':10,'success':2,'avg_ms':9000,'score':1}
    monitor.collect()
    assert support.model_runtime._monitor_penalties['billing:primary']>0
    assert monitor.snapshot()['active_alerts']
    assert list(tmp_path.glob('alerts-*.jsonl'))
    support.model_runtime._health['billing:primary']={'total':11,'success':11,'avg_ms':500,'score':1}
    monitor.collect()
    assert not monitor.snapshot()['active_alerts']

def test_zscore_uses_prior_window_not_current_outlier():
    detector=AnomalyDetector()
    for i in range(30): detector.record('latency',i%2)
    assert detector.record('latency',100)>2.5

def test_failed_judge_is_unscored_and_rejected_usage_is_not_lost(monkeypatch):
    from servemind.llm.deepseek_client import Completion
    class BadJudge:
        def complete(self,**kwargs): return Completion('not JSON','fixture',20,10)
    monkeypatch.setattr('servemind.evaluation.evaluator.DeepSeekClient',BadJudge)
    report=evaluate_curated_commerce([CURATED_COMMERCE_CASES[0]],live_judge=True)
    assert report['live_judge']['mean'] is None
    assert report['live_judge']['success_rate']==0
    assert report['live_judge']['actual_input_tokens']==20

def test_operations_api_is_private_and_readiness_checks_dependencies(tmp_path):
    from fastapi.testclient import TestClient
    from servemind.api.main import create_app
    app=create_app(commerce_db_path=tmp_path/'test.sqlite',use_llm=False)
    with TestClient(app) as client:
        assert client.get('/metrics').status_code==200
        assert client.get('/v1/operations').status_code==200
        app.state.commerce_support.working_memory.client=SimpleNamespace(ping=lambda:False)
        assert client.get('/v1/ready').status_code==503
    with TestClient(app,client=('203.0.113.2',5000)) as client:
        assert client.get('/metrics').status_code==403
