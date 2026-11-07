"""Shared evidence scope, response completeness and factual prose contracts."""
from __future__ import annotations

import copy
import json
import math


def completion_status(response):
    """CAMEL uses termination_reasons; the provider API uses finish_reason."""
    info = getattr(response, 'info', None) or {}
    reasons = info.get('termination_reasons', info.get('finish_reasons', []))
    if isinstance(reasons, str):
        reasons = [reasons]
    if not reasons:
        reasons = [getattr(c, 'finish_reason', None) for c in getattr(response, 'choices', [])]
    reasons = [str(x).lower() for x in reasons if x is not None]
    incomplete = any(x in ('length', 'max_tokens', 'max_output_tokens', 'content_filter')
                     or ('token' in x and any(word in x for word in ('max', 'limit', 'exceed'))) for x in reasons)
    return {'complete': not incomplete, 'finish_reasons': reasons,
            'status': 'incomplete' if incomplete else 'complete'}


def resolve_scoped_evidence(records):
    """Resolve explicit record relations within one case/event, never by recency alone."""
    items = copy.deepcopy(records)
    by_id = {}
    for item in items:
        key = item.get('evidence_id')
        if not isinstance(key, str) or not key or key in by_id:
            raise ValueError('Evidence IDs must be nonempty and unique')
        if item.get('scope') not in ('case', 'global_reference'):
            raise ValueError('Evidence scope must be explicit')
        if item['scope'] == 'case' and not item.get('case_id'):
            raise ValueError('Case evidence requires case_id')
        if not isinstance(item.get('supersedes', []), list):
            raise ValueError('Evidence supersedes must be a list')
        by_id[key] = item
    edges = {}
    for item in items:
        targets = item.get('supersedes', [])
        if targets and item['scope'] != 'case':
            raise ValueError('Global references cannot supersede case evidence')
        for target_id in targets:
            target = by_id.get(target_id)
            if target is None or target_id == item['evidence_id']:
                raise ValueError('Invalid evidence relation')
            if target['scope'] != 'case' or any(item.get(k) != target.get(k) for k in ('case_id', 'event_id')):
                raise ValueError('Cross-case or cross-event evidence replacement is forbidden')
            if item.get('accepts_sha256') is not None and item['accepts_sha256'] != target.get('content_sha256'):
                raise ValueError('Accepted reference hash does not match the scoped source')
        edges[item['evidence_id']] = targets
    def visit(key, path):
        if key in path:
            raise ValueError('Evidence replacement cycle')
        for child in edges[key]:
            visit(child, path | {key})
    for key in edges:
        visit(key, set())
    superseded = {x for values in edges.values() for x in values}
    active = [x for x in items if x['evidence_id'] not in superseded]
    grouped = {}
    for item in active:
        grouped.setdefault(str(item.get('case_id') or 'global_reference'), []).append(item)
    return {'active': active, 'by_case': grouped,
            'history': [x for x in items if x['evidence_id'] in superseded]}


def evidence_context(case_id=None, request_id=None, capture_times=None, records=None):
    resolved = resolve_scoped_evidence(records or [])
    scope_id = str(request_id or case_id or 'unassigned')
    foreign = [x['evidence_id'] for x in resolved['active']
               if x['scope'] == 'case' and str(x['case_id']) != str(case_id or scope_id)]
    if foreign:
        raise ValueError('Evidence belongs to another case: ' + ', '.join(foreign))
    return {'case_id': str(case_id) if case_id is not None else None,
            'request_id': str(request_id) if request_id is not None else None,
            'scope_id': scope_id, 'capture_times': list(capture_times or []),
            'evidence': resolved,
            'rules': ['量测仅属于当前请求和案例，采集时相保持原样。',
                      '知识和外部检索图只作为参考；未验证属于当前案例时不得继承其量测。',
                      '来源状态仅在同案例同事件且有明确替代或哈希接受关系时合并；旧快照保留。']}


def context_text(context):
    return '本次证据范围与记录关系：\n' + json.dumps(context, ensure_ascii=False, separators=(',', ':'))


def _different(a, b):
    if isinstance(b, dict):
        return not isinstance(a, dict) or set(a) != set(b) or any(_different(a[k], b[k]) for k in b)
    if isinstance(b, bool):
        return type(a) is not bool or a != b
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        return (not isinstance(a, (int, float)) or isinstance(a, bool)
                or not math.isfinite(a) or abs(a - b) > .01)
    return a != b


def render_fact_summary(facts, labels=None):
    """Render from trusted data, including explicit missing fields and nested units."""
    labels = labels or {}
    rows = []
    def add(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                add(child, path + '.' + key if path else key)
            return
        if value is None:
            text = '未提供'
        elif isinstance(value, bool):
            text = '是' if value else '否'
        elif isinstance(value, float):
            text = f'{value:.6f}'.rstrip('0').rstrip('.')
        elif isinstance(value, list):
            text = '、'.join(str(x) if x is not None else '未提供' for x in value)
        else:
            text = str(value)
        rows.append(f"{labels.get(path, path)}：{text}")
    add(facts, '')
    return '本次可核对事实\n' + '\n'.join(rows)


def bind_structured_report(original, trusted_facts, labels=None, evidence_records=None, case_id=None):
    """When facts change, never deliver the stale generated conclusion beside them."""
    original = copy.deepcopy(original) if isinstance(original, dict) else {}
    original_facts = original.get('generated_facts_before_check', original.get('facts'))
    changed = _different(original_facts, trusted_facts)
    result = copy.deepcopy(original)
    result['facts'] = copy.deepcopy(trusted_facts)
    result['fact_source'] = 'authoritative_tool_fields'
    result['raw_generated_report'] = original
    scope_errors = []
    if evidence_records is not None:
        resolved = resolve_scoped_evidence(evidence_records)
        by_id = {x['evidence_id']:x for x in resolved['active']}
        cited = original.get('evidence_ids', original.get('citations', []))
        if not isinstance(cited,list) or any(not isinstance(key,str) or key not in by_id for key in cited):
            scope_errors.append('unresolved_or_superseded_citation')
        else:
            cases = {str(by_id[key]['case_id']) for key in cited if by_id[key]['scope']=='case'}
            if len(cases)>1:
                scope_errors.append('mixed_case_citations_require_separate_reports')
            if case_id is not None and any(key != str(case_id) for key in cases):
                scope_errors.append('foreign_case_citation')
    result['source_scope_errors'] = scope_errors
    if changed or scope_errors:
        result['conclusion'] = render_fact_summary(trusted_facts, labels)
        result['interpretation_status'] = 'withheld_after_scope_check' if scope_errors else 'withheld_after_fact_correction'
        result['review_required'] = True
    else:
        result['interpretation_status'] = 'generated_requires_scope_and_semantic_review'
        result['review_required'] = False
    return result
