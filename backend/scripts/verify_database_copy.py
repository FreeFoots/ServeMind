"""Read-only verification before switching a stopped project's database URL."""
import argparse
import hashlib
import json
import os
from datetime import datetime

import psycopg
from psycopg import sql

from servemind.config.settings import PROJECT_ROOT
from servemind.service.commerce_migration import TABLES


def signature(connection, table):
    columns = [r[0] for r in connection.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='commerce' AND table_name=%s ORDER BY ordinal_position",(table,))]
    keys = [r[0] for r in connection.execute("SELECT k.column_name FROM information_schema.table_constraints t JOIN information_schema.key_column_usage k USING(constraint_catalog,constraint_schema,constraint_name) WHERE t.table_schema='commerce' AND t.table_name=%s AND t.constraint_type='PRIMARY KEY' ORDER BY k.ordinal_position",(table,))]
    query = sql.SQL('SELECT {} FROM commerce.{} ORDER BY {}').format(sql.SQL(',').join(map(sql.Identifier,columns)),sql.Identifier(table),sql.SQL(',').join(map(sql.Identifier,keys)))
    digest = hashlib.sha256()
    count = 0
    for row in connection.execute(query):
        digest.update(json.dumps([v.isoformat(timespec='microseconds') if isinstance(v,datetime) else v for v in row],ensure_ascii=False,separators=(',',':')).encode()+b'\n')
        count += 1
    return {'rows':count,'sha256':digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',default=os.getenv('SERVEMIND_DATABASE_URL','postgresql:///servemind'))
    parser.add_argument('--target',required=True,help='PostgreSQL connection URI without password; use peer authentication')
    args = parser.parse_args()
    source = args.source
    if source == args.target:
        parser.error('source and target must be different database instances')
    with psycopg.connect(source,options='-c timezone=UTC') as before, psycopg.connect(args.target,options='-c timezone=UTC') as after:
        hashes = {t:{'source':signature(before,t),'target':signature(after,t)} for t in TABLES}
        tables = before.execute("SELECT table_schema,table_name FROM information_schema.tables WHERE table_schema IN ('commerce','msom','memory','knowledge') AND table_type='BASE TABLE' ORDER BY table_schema,table_name").fetchall()
        counts = {}
        for schema,table in tables:
            query = sql.SQL('SELECT count(*) FROM {}.{}').format(sql.Identifier(schema),sql.Identifier(table))
            counts[schema+'.'+table]={'source':before.execute(query).fetchone()[0],'target':after.execute(query).fetchone()[0]}
        report = {'passed':all(v['source']==v['target'] for v in hashes.values()) and all(v['source']==v['target'] for v in counts.values()),
                  'business_hashes':hashes,'table_counts':counts,
                  'target_data_directory':after.execute('SHOW data_directory').fetchone()[0]}
    output = PROJECT_ROOT/'backend/runtime/database-copy-verification.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(output),'passed':report['passed'],'tables_verified':len(counts),'business_hashes_verified':len(hashes)},ensure_ascii=False),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
