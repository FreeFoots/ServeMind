"""Run only while the SQLite-backed API is stopped; keep both source and backup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from servemind.config.settings import COMMERCE_DB_PATH, PROJECT_ROOT
from servemind.service.commerce_migration import migrate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=COMMERCE_DB_PATH)
    parser.add_argument('--offline-confirmed',action='store_true',help='Confirm the source-writing API is stopped')
    args = parser.parse_args()
    if not args.offline_confirmed:
        parser.error('stop the SQLite API, then pass --offline-confirmed')
    report = migrate(args.source)
    output = PROJECT_ROOT/'backend/runtime/commerce-migration.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(output),'already_applied':report['already_applied'],
        'verified':report.get('verification',{}).get('passed'),'backup':report.get('backup'),
        'counts':{t:c['source']['rows'] for t,c in report.get('verification',{}).get('tables',{}).items()}},ensure_ascii=False))


if __name__ == '__main__':
    main()
