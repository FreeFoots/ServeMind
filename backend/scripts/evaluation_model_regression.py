"""Evaluate the actual intent/domain-routing/specialist path, not historical rule scores.

Explicit --live enables paid generation/Judge. Reports never update a baseline.
Small samples and missing human annotations always block release.
"""
import argparse
import hashlib
import json
from pathlib import Path

from servemind.config.settings import PROJECT_ROOT
from servemind.evaluation.curated_cases import CURATED_COMMERCE_CASES
from servemind.evaluation.evaluator import evaluate_curated_commerce, evaluate_model_intents, evaluate_policy_rag
from servemind.evaluation.test_cases import INTENT_CASES, DATA_INTENT_CASES
from servemind.evaluation.expanded_cases import EXPANDED_INTENT_CASES
from servemind.core.knowledge_base import KnowledgeBase
from servemind.evaluation.metrics import baseline_gate, wilson_lower
from servemind.config.model_settings import model_metadata

def build_report(cases, *, live=False, baseline=None, intent_limit=0):
    evaluation=evaluate_curated_commerce(cases,live_generation=live,live_judge=live)
    runtime=evaluation['runtime_metrics']
    metrics={'dialogue_accuracy':evaluation['accuracy'],
             'task_completion':runtime['task_completion_rate'],
             'specialist_success':runtime['specialist_model_success_rate'],
             'judge_quality':evaluation['live_judge']['mean'] or 0}
    checks={'sample_size':len(cases)>=24,'human_annotations':evaluation['human_review_gate']['passed'],
            'real_model':live and runtime['model_agent_requests']>0,
            'specialist_success':metrics['specialist_success']>=.9,
            'dialogue_accuracy':evaluation['accuracy']>=.9,
            'wilson_lower_bound':wilson_lower(evaluation['passed'],evaluation['total'])>=.75,
            'task_completion':metrics['task_completion']>=.95,
            'judge_coverage':evaluation['live_judge']['success_rate']==1,
            'judge_quality':metrics['judge_quality']>=.85,
            'tool_scope':runtime['unauthorized_block_count']==0,
            'grounding':runtime['grounding_failure_count']==0,
            'intent_recognition':runtime['intent_recognition_fallback_count']==0}
    report={'protocol':'commercial-domain-routing-regression-v3',
            'dataset_sha256':hashlib.sha256(json.dumps(cases,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
            'model':model_metadata()['model'], 'execution_scope':evaluation['execution_scope'],
            'metrics':metrics,'evaluation':evaluation,
            'wilson_95_lower':wilson_lower(evaluation['passed'],evaluation['total'])}
    knowledge=KnowledgeBase()
    report['knowledge_version']=knowledge.corpus_version
    report['policy_rag']=evaluate_policy_rag(knowledge)
    checks['policy_rag']=report['policy_rag']['accuracy']==1
    if intent_limit:
        if not live:
            raise ValueError('real_intents_require_live')
        from servemind.service.commerce_support import CommerceSupport
        report['model_intent']=evaluate_model_intents(CommerceSupport(),
            [*INTENT_CASES,*DATA_INTENT_CASES,*EXPANDED_INTENT_CASES][:intent_limit])
        report['metrics']['intent_macro_f1']=report['model_intent']['classification']['macro_f1']
        checks['model_intent_f1']=report['metrics']['intent_macro_f1']>=.85
    if baseline is not None:
        report['baseline']=baseline_gate(report,baseline)
        checks['no_regression']=report['baseline']['passed']
    report['release_gate']={'passed':all(checks.values()),'checks':checks}
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--limit',type=int,default=24)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--intent-limit',type=int,default=0,help='Explicit real intent recognition sample count (0 skips; maximum 400)')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'backend/runtime/model-regression.json')
    args=parser.parse_args()
    if not 1<=args.limit<=len(CURATED_COMMERCE_CASES):
        parser.error('limit outside curated dataset')
    if not 0<=args.intent_limit<=400 or args.intent_limit and not args.live:
        parser.error('intent-limit requires --live and a value between 0 and 400')
    if args.baseline and args.output.resolve()==args.baseline.resolve():
        parser.error('output must not overwrite baseline')
    report=build_report(CURATED_COMMERCE_CASES[:args.limit],live=args.live,
                        baseline=json.loads(args.baseline.read_text()) if args.baseline else None,intent_limit=args.intent_limit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    args.output.chmod(0o600)
    print(json.dumps({'output':str(args.output),'metrics':report['metrics'],'release_gate':report['release_gate']},ensure_ascii=False))
    return 0 if report['release_gate']['passed'] else 1

if __name__=='__main__':
    raise SystemExit(main())
