"""Opt-in native PostgreSQL integration; small test schemas are retained, not dropped."""
import os
import uuid
from contextlib import contextmanager
from decimal import Decimal

import psycopg
import pytest
from fastapi import HTTPException
from psycopg import sql
from psycopg.rows import dict_row

from servemind.config import settings  # loads backend/.env without printing secrets
from servemind.service.commerce_migration import migrate
from servemind.service.commerce_store import CommerceStore
from servemind.service.commerce_support import CommerceSupport
from servemind.service.postgres_commerce_store import PostgresCommerceStore, bound_sql
from servemind.service.postgres_commerce_store import PostgresConversationLock
from servemind.memory.postgres_memory import PostgresConversationMemory


def test_bound_sql_keeps_literal_question_and_bound_values_separate():
    assert bound_sql("SELECT '?' AS label WHERE id=? AND label LIKE '%foo%'") == "SELECT '?' AS label WHERE id=%s AND label LIKE '%%foo%%'"


@pytest.fixture
def pg(monkeypatch):
    if os.getenv('SERVEMIND_TEST_POSTGRES') != 'true':
        pytest.skip('enable SERVEMIND_TEST_POSTGRES for the local integration database')
    monkeypatch.setenv('SERVEMIND_RAG_BACKEND', 'keyword')
    monkeypatch.setenv('SERVEMIND_SEMANTIC_INTENT', 'false')
    return os.getenv('SERVEMIND_DATABASE_URL', 'postgresql:///servemind'), 'test_commerce_'+uuid.uuid4().hex


def account(store, name, role):
    result = store.register(username=name, display_name=name, role=role, password='test-password-123')
    return result['account'], result['token']


def test_native_postgres_workflow_authorization_and_idempotence(pg):
    dsn, schema = pg
    store = PostgresCommerceStore(dsn, schema=schema, require_migration=False)
    merchant, _ = account(store, 'pg-merchant', 'merchant')
    buyer, token = account(store, 'pg-buyer', 'buyer')
    stranger, _ = account(store, 'pg-stranger', 'buyer')
    assert store.authenticate(token)['id'] == buyer['id']
    product = store.create_product(account=merchant, title='耳机', description='无线', sku_id=None, price=Decimal('99'))
    conv = store.create_conversation(account=buyer, product_id=product['id'])
    support = CommerceSupport(use_llm=False)
    kwargs = dict(account=buyer, conversation_id=conv['id'], content='价格多少', client_message_id='p1', support=support)
    first = store.send_message(**kwargs)
    assert '99.00' in first['ai_message']['content']
    assert first == store.send_message(**kwargs)
    with pytest.raises(HTTPException) as error:
        store.get_conversation(account=merchant, conversation_id=conv['id'])
    assert error.value.status_code == 404
    with pytest.raises(HTTPException):
        store.get_conversation(account=stranger, conversation_id=conv['id'])
    shared = store.send_message(account=buyer, conversation_id=conv['id'], content='请转人工给商家', client_message_id='p2', support=support)
    assert shared['ai_message']['metadata']['needs_merchant']
    assert store.get_conversation(account=merchant, conversation_id=conv['id'])['merchant_visible']
    reply = store.send_message(account=merchant, conversation_id=conv['id'], content='您好，我来处理', client_message_id='m1', support=support)
    assert reply['message']['sender_type'] == 'merchant' and reply['ai_message'] is None
    with psycopg.connect(dsn) as connection:
        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute(sql.SQL('UPDATE {}.products SET display_price_cents=0 WHERE id=%s').format(sql.Identifier(schema)), (product['id'],))


def test_snapshot_migration_preserves_credentials_and_messages(pg, tmp_path):
    dsn, schema = pg
    source = tmp_path/'commerce.sqlite3'
    original = CommerceStore(source)
    merchant, _ = account(original, 'snapshot-merchant', 'merchant')
    buyer, token = account(original, 'snapshot-buyer', 'buyer')
    product = original.create_product(account=merchant, title='母婴1', description='', sku_id=None, price=Decimal('10'))
    conv = original.create_conversation(account=buyer, product_id=product['id'])
    support = CommerceSupport(use_llm=False)
    args = dict(account=buyer, conversation_id=conv['id'], content='多少钱', client_message_id='stable-id', support=support)
    before = original.send_message(**args)
    report = migrate(source, dsn=dsn, schema=schema, backup_root=tmp_path/'backups')
    assert report['verification']['passed']
    assert all(c['matched'] for c in report['verification']['tables'].values())
    target = PostgresCommerceStore(dsn, schema=schema)
    assert target.authenticate(token)['id'] == buyer['id']
    assert target.send_message(**args) == before
    assert target.login(username='snapshot-buyer', password='test-password-123')['account']['id'] == buyer['id']
    assert migrate(source, dsn=dsn, schema=schema, backup_root=tmp_path/'backups')['already_applied']
    subsequent = target.send_message(**{**args, 'content':'库存呢', 'client_message_id':'new-id'})
    assert subsequent['ai_message']['id'] != before['ai_message']['id']
    with psycopg.connect(dsn) as connection:
        assert connection.execute(sql.SQL('SELECT count(*) FROM {}.messages').format(sql.Identifier(schema))).fetchone()[0] == 4


def test_memory_failure_rolls_back_reply_and_handoff_and_retry_is_idempotent(pg):
    dsn, schema = pg
    store = PostgresCommerceStore(dsn,schema=schema,require_migration=False)
    merchant, _ = account(store,'atomic-merchant','merchant')
    buyer, _ = account(store,'atomic-buyer','buyer')
    product = store.create_product(account=merchant,title='电子产品1',description='',sku_id=None,price=Decimal('88'))
    conv = store.create_conversation(account=buyer,product_id=product['id'])
    memory = PostgresConversationMemory(dsn,schema='test_memory_'+uuid.uuid4().hex)
    class Cache:
        writes = []
        def get(self,*args): return {}
        def put(self,*args): self.writes.append(args); return True
    cache = Cache()
    support = CommerceSupport(use_llm=False,working_memory=cache)
    support.durable_memory = memory
    support.semantic_memory = False
    write = memory.record_exchange
    def broken(**kwargs): raise PermissionError('simulated_memory_failure')
    memory.record_exchange = broken
    args = dict(account=buyer,conversation_id=conv['id'],content='请转人工给商家',client_message_id='atomic',support=support)
    with pytest.raises(PermissionError): store.send_message(**args)
    assert not cache.writes
    actual = store.get_conversation(account=buyer,conversation_id=conv['id'])
    assert len(actual['messages']) == 1 and not actual['merchant_visible']
    assert actual['handoff_state']=='ai_support'
    memory.record_exchange = write
    result = store.send_message(**args)
    assert result['ai_message']['metadata']['needs_merchant']
    assert '_memory_prepared' not in result['ai_message']['metadata']
    assert len(cache.writes)==1
    assert store.send_message(**args) == result
    ctx = memory.context(conv['id'],buyer['id'],merchant['id'],product['id'])
    assert ctx['turn_count']==1 and ctx['profile']['interaction_count']==1


def test_semantic_memory_filters_scope_and_expiry_before_ranking(pg, monkeypatch):
    dsn, _ = pg
    memory = PostgresConversationMemory(dsn,schema='test_memory_'+uuid.uuid4().hex)
    vector = [1.0]+[0.0]*1023
    monkeypatch.setattr('servemind.memory.postgres_memory.bounded_embedding',lambda *args,**kwargs:vector)
    cases = [('safe','buyer','merchant','product'),('other-buyer','stranger','merchant','product'),
             ('other-merchant','buyer','stranger','product'),('other-product','buyer','merchant','other'),
             ('expired','buyer','merchant','product')]
    for cid,bid,mid,pid in cases:
        memory.record_exchange(conversation_id=cid,buyer_id=bid,merchant_id=mid,product_id=pid,
            message_id='message-'+cid,intent='refund_policy',topics=['refund_policy'],evidence_ids=['ev'],
            needs_merchant=True,pending_topics=['refund_policy'],
            prepared={'summary':cid,'embedding':vector,'source':'structured_topics'})
    with psycopg.connect(dsn) as connection:
        connection.execute(sql.SQL("UPDATE {}.episodes SET created_at=now()-interval '31 days' WHERE conversation_id='expired'").format(sql.Identifier(memory.schema)))
    result = memory.context('new-session','buyer','merchant','product',query='上次的退款',semantic=True)
    assert [e['conversation_id'] for e in result['episodes']]==['safe']
    assert not memory.context('safe','buyer','merchant','wrong-product')['summary']
    with pytest.raises(PermissionError):
        memory.record_exchange(conversation_id='safe',buyer_id='stranger',merchant_id='merchant',product_id='product',
            message_id='forged',intent='other',topics=[],evidence_ids=[],needs_merchant=False)


def test_postgres_lock_works_without_redis(pg):
    dsn, _ = pg
    class NoRedis:
        @contextmanager
        def hold(self,*args,**kwargs): yield 'unavailable'
    first, second = PostgresConversationLock(dsn), PostgresConversationLock(dsn)
    first.fast = second.fast = NoRedis()
    cid = 'lock-'+uuid.uuid4().hex
    with first.hold(cid):
        with pytest.raises(TimeoutError):
            with second.hold(cid,wait_seconds=.1): pass
    with second.hold(cid,wait_seconds=.1): pass
