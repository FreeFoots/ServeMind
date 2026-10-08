"""Classification metrics and comparable, immutable-baseline regression checks."""
from __future__ import annotations

import math

def classification_metrics(expected: list[str], predicted: list[str]) -> dict:
    if len(expected)!=len(predicted):
        raise ValueError('classification_length_mismatch')
    labels = sorted(set(expected)|set(predicted))
    rows, matrix = {}, {label:{other:0 for other in labels} for label in labels}
    for want, got in zip(expected,predicted):
        matrix[want][got]+=1
    for label in labels:
        tp=matrix[label][label]
        fp=sum(matrix[other][label] for other in labels if other!=label)
        fn=sum(matrix[label][other] for other in labels if other!=label)
        p=tp/(tp+fp) if tp+fp else 0
        r=tp/(tp+fn) if tp+fn else 0
        rows[label]={'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0,
                     'support':sum(matrix[label].values())}
    return {'accuracy':sum(a==b for a,b in zip(expected,predicted))/len(expected) if expected else 0,
            'macro_precision':sum(r['precision'] for r in rows.values())/len(rows) if rows else 0,
            'macro_recall':sum(r['recall'] for r in rows.values())/len(rows) if rows else 0,
            'macro_f1':sum(r['f1'] for r in rows.values())/len(rows) if rows else 0,
            'per_class':rows,'confusion_matrix':matrix}

def wilson_lower(passed: int,total: int) -> float:
    if not total:
        return 0
    z=1.96
    p=passed/total
    return (p+z*z/(2*total)-z*math.sqrt((p*(1-p)+z*z/(4*total))/total))/(1+z*z/total)

def baseline_gate(current: dict, baseline: dict, tolerance=.03) -> dict:
    # A legacy/proxy total is never compared to a real-model business score.
    compatible = all(current.get(k) is not None and current.get(k)==baseline.get(k)
                     for k in ('protocol','dataset_sha256','model','execution_scope'))
    if 'knowledge_version' in current:
        compatible = compatible and current['knowledge_version']==baseline.get('knowledge_version')
    old,new = baseline.get('metrics',{}),current.get('metrics',{})
    comparable = set(old)&set(new)
    checks={k:isinstance(new[k],(int,float)) and isinstance(old[k],(int,float))
            and new[k]>=old[k]-tolerance for k in comparable}
    return {'passed':compatible and bool(checks) and all(checks.values()),
            'compatible':compatible,'max_absolute_degradation':tolerance,'checks':checks}
