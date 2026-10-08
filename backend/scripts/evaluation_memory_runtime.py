"""Real model + PostgreSQL + Qwen memory smoke in a retained isolated test scope."""
import argparse
import json
import os
import uuid
from decimal import Decimal

from servemind.config.settings import PROJECT_ROOT
from servemind.memory.postgres_memory import PostgresConversationMemory
from servemind.service.commerce_support import CommerceSupport
from servemind.service.postgres_commerce_store import PostgresCommerceStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true',help='Call the configured model; paid synthetic smoke, not release approval')
    parser.add_argument('--ambiguous-followup',action='store_true',help='Diagnose a pronoun-only follow-up with actual persistent conversation context')
    parser.add_argument('--output',type=__import__('pathlib').Path,default=PROJECT_ROOT/'backend/runtime/memory-runtime-smoke.json')
    args = parser.parse_args()
    if not args.live:
        parser.error('pass --live to use the configured provider')
    dsn = os.getenv('SERVEMIND_DATABASE_URL','postgresql:///servemind')
    suffix = uuid.uuid4().hex
    store = PostgresCommerceStore(dsn,schema='test_commerce_'+suffix,require_migration=False)
    memory = PostgresConversationMemory(dsn,schema='test_memory_'+suffix)
    merchant = store.register(username='memory-smoke-merchant',display_name='记忆测试商家',role='merchant',password='synthetic-test-password')['account']
    buyer = store.register(username='memory-smoke-buyer',display_name='记忆测试买家',role='buyer',password='synthetic-test-password')['account']
    product = store.create_product(account=merchant,title='耳机',description='无线耳机',sku_id=None,price=Decimal('99'))
    conv = store.create_conversation(account=buyer,product_id=product['id'])
    support = CommerceSupport()
    if not support.model_runtime:
        raise RuntimeError('real_model_runtime_required')
    support.durable_memory, support.semantic_memory = memory, True
    records = []
    questions = (('商品还有货吗','那它呢？继续说刚才的') if args.ambiguous_followup else
                 ('还有货吗，价格多少？','刚才的价格和库存，再简单说一次。'))
    for i, question in enumerate(questions):
        response = store.send_message(account=buyer,conversation_id=conv['id'],content=question,
                                      client_message_id=f'memory-{i}',support=support)
        meta = response['ai_message']['metadata']
        records.append({'question':question,'answer':response['ai_message']['content'], 'topics':meta['topics'],
            'model_use':meta['model_use'],'memory_processing':meta.get('memory_processing'),
            'semantic_memory_recalled':meta.get('semantic_memory_recalled'),
            'episodic_memory_used':meta['episodic_memory_used'],
            'grounding':meta['grounding'],'calls':meta['agent_runtime']['calls']})
    cross_session = memory.context('new-synthetic-session',buyer['id'],merchant['id'],product['id'],
                                   query='之前咨询商品价格和库存',semantic=True)
    wrong_merchant = memory.context('new-synthetic-session',buyer['id'],'other-merchant',product['id'],
                                    query='之前咨询商品价格和库存',semantic=True)
    checks = {'model_summaries':all(r['memory_processing']['model_succeeded'] for r in records),
              'indexed_episodes':all(r['memory_processing']['embedding_ready'] for r in records),
              'second_turn_context':records[1]['episodic_memory_used'],
              'cross_session_recall':bool(cross_session['episodes']),
              'merchant_isolation':not wrong_merchant['episodes'],
              'grounded':all(r['grounding']['passed'] for r in records),
              'durable_turn_count':memory.context(conv['id'],buyer['id'],merchant['id'],product['id'])['turn_count']==2}
    if args.ambiguous_followup:
        checks['ambiguous_followup_inventory'] = ('inventory_query' in records[1]['topics'] and
                                                '在售' in records[1]['answer'])
    report = {'kind':'real_provider_memory_smoke_not_release','checks':checks,'passed':all(checks.values()),
              'scenario':'ambiguous_followup' if args.ambiguous_followup else 'explicit_price_stock_followup',
              'commerce_schema':store.schema,'memory_schema':memory.schema,'turns':records,
              'cross_session_recall_count':len(cross_session['episodes']),
              'human_reviewed':False,'release_gate_passed':False}
    output = args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    output.chmod(0o600)
    print(json.dumps({'report':str(output),'checks':checks,'passed':report['passed']},ensure_ascii=False),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
