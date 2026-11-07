"""Portable paired reports from a cached inference, with no inference or LLM calls."""
import base64
import copy
import html
import io
import json
import math
from urllib.parse import urlsplit

from PIL import Image


def _public(value):
    if isinstance(value, dict):
        private = {'api_key', 'authorization', 'password', 'token', 'image_path',
                   'overlay_b64', 'image_b64', 'content_b64'}
        return {str(k): _public(v) for k, v in value.items()
                if not str(k).startswith('_') and str(k).lower() not in private
                and not str(k).lower().endswith('_b64')}
    if isinstance(value, (tuple, list)):
        return [_public(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value if value is None or isinstance(value, (str, int, float, bool)) else None


def comparison_record(result):
    record = result.get('review_handoff')
    if result.get('region') != 'compare' or not isinstance(record, dict):
        raise ValueError('请使用已生成的双期变化报告')
    if record.get('summary', {}).get('kind') != 'paired_mask_change':
        raise ValueError('本次结果不是双期变化报告')
    if not record.get('trace', {}).get('trace_complete'):
        raise ValueError('本次结果缺少完整请求与权重标识，请重新生成后导出')
    record = _public(copy.deepcopy(record))
    # Provider error text is for runtime diagnostics, not a portable report.
    record['automatic_analysis'] = {
        'generation': {k: (result.get('generation') or {}).get(k) for k in ('status', 'source')},
        'degraded': bool((result.get('quality') or {}).get('degraded')),
        'answer': result.get('final_answer') or '',
    }
    return {
        'schema_version': 1, 'report_type': 'paired_optical_screening',
        'report_id': result['report_id'], 'generated_at': result.get('generated_at'),
        'objective_facts_source': 'same_detector_result',
        'objective_facts': _public(result.get('objective_facts') or record['objective_facts']),
        'review_handoff': record,
        'illustrations_included_in_json': False,
    }


def _text(value):
    return html.escape('未提供' if value is None else str(value), quote=True)


def _json(value):
    return _text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _link(value):
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value)
        if parts.scheme in ('https', 'http') and parts.hostname and not parts.username and not parts.password:
            return value
    except ValueError:
        pass
    return None


def _image(value):
    """Accept only verified bounded PNG/JPEG bytes, never SVG or arbitrary data URLs."""
    if not isinstance(value, str) or not value or len(value) > 32 * 1024 * 1024:
        return None
    try:
        raw = base64.b64decode(value, validate=True)
        if not (raw.startswith(b'\x89PNG\r\n\x1a\n') or raw.startswith(b'\xff\xd8\xff')):
            return None
        with Image.open(io.BytesIO(raw), formats=['PNG', 'JPEG']) as picture:
            if picture.format not in ('PNG', 'JPEG'):
                return None
            mime = 'image/png' if picture.format == 'PNG' else 'image/jpeg'
            picture.verify()
        return 'data:' + mime + ';base64,' + value
    except (ValueError, OSError, Image.DecompressionBombError):
        return None


def comparison_html(result, packet=None):
    packet = comparison_record(result) if packet is None else packet
    record = packet['review_handoff']
    facts = packet['objective_facts']; change = facts.get('change') or {}
    trace = record['trace']; evidence = record['input_evidence']
    measurements = (
        ('T1 掩膜占比', change.get('t1_percent'), '%'),
        ('T2 掩膜占比', change.get('t2_percent'), '%'),
        ('净变化', change.get('net_percentage_points'), ' 个百分点'),
        ('新增差分连通区域', change.get('added_regions'), ' 处'),
        ('扩大差分连通区域', change.get('expanded_regions'), ' 处'),
        ('消退差分连通区域', change.get('removed_regions'), ' 处'),
    )
    cards = ''.join('<div class="metric"><span>' + _text(label) + '</span><strong>'
                    + _text(value) + (unit if value is not None else '') + '</strong></div>'
                    for label, value, unit in measurements)
    illustrations = []
    det = result['detection']
    for field, label in (('t1_overlay_b64', 'T1 预测叠加图'), ('t2_overlay_b64', 'T2 预测叠加图'),
                         ('overlay_b64', '共同有效范围内的变化叠加图')):
        src = _image(det.get(field))
        illustrations.append('<figure><figcaption>' + label + '</figcaption>'
                             + ('<img src="' + src + '" alt="' + label + '">' if src else '<p>本次结果未提供可用图像</p>')
                             + '</figure>')
    checks = ''.join('<li>' + _text(c.get('task')) + ('（' + _text(c['epoch']) + '）' if c.get('epoch') else '')
                     + '</li>' for c in record.get('pending_checks', []))
    sources = []
    for source in record.get('knowledge_sources', []):
        title = _text(source.get('id', '')) + ' ' + _text(source.get('title'))
        url = _link(source.get('source_url'))
        sources.append('<li>' + ('<a href="' + _text(url) + '" rel="noopener noreferrer">' + title + '</a>' if url else title)
                       + '</li>')
    times = ''.join('<tr><th>' + epoch.upper() + ' 采集时间</th><td>'
                    + _text(evidence.get(epoch, {}).get('acquisition_datetime')) + '</td></tr>' for epoch in ('t1', 't2'))
    trace_rows = [('模型标识', trace.get('model_tag')), ('掩膜阈值', trace.get('threshold')),
                  ('模型权重 SHA-256', '\n'.join(trace.get('weights_sha256') or []) or None),
                  ('T1 原始输入 SHA-256', evidence.get('t1', {}).get('input_sha256')),
                  ('T2 原始输入 SHA-256', evidence.get('t2', {}).get('input_sha256'))]
    trace_table = '<table><tbody>' + ''.join('<tr><th>' + _text(label) + '</th><td>'
                                           + _text(value) + '</td></tr>' for label, value in trace_rows) + '</tbody></table>'
    items = record.get('review_items', [])
    decisions = record.get('accepted_human_review_decisions') or []
    review_rows = ''.join('<li><b>' + _text(row.get('status_label')) + '</b> · ' + _text(row.get('target_key'))
                         + ' · ' + _text(row.get('reviewer')) + ' · ' + _text(row.get('created_at'))
                         + '<br>' + _text(row.get('note')) + '<br>依据：' + _text(row.get('evidence')) + '</li>' for row in decisions)
    review_state = '已记录人工意见，独立核查条件仍待确认' if decisions else '尚未记录有效人工意见'
    item_rows = ''.join('<tr><td>' + _text(item.get('review_item_key')) + '</td><td>'
                        + _text(item.get('change_type')) + '</td><td>' + _text(item.get('area_px'))
                        + '</td><td>' + _text(item.get('area_m2')) + '</td><td>'
                        + ('已记录人工意见' if item.get('review_status') == 'human_note_recorded' else '待复核') + '</td></tr>' for item in items)
    item_table = ('<table><thead><tr><th>期别与区域</th><th>模型匹配变化类型</th><th>面积（像素）</th>'
                  '<th>面积（平方米）</th><th>复核状态</th></tr></thead><tbody>' + item_rows + '</tbody></table>') if items else '<p>本次结果未提供可交接的匹配区域明细。</p>'
    generation = record['automatic_analysis']['generation']
    status = '检测事实汇总（自动分析未完成）' if generation.get('status') == 'fallback' else '自动分析结果（待人工复核）'
    appendix = {'trace': trace, 'input_evidence': evidence, 'observation_support': record['observation_support'],
                'registration': record['registration'], 'measurement_scale': record['measurement_scale'],
                'summary': record['summary'],
                'review_items': items, 'review_items_truncated': record['review_items_truncated'],
                'objective_facts': facts}
    return '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>矿安天眼 · 双期影像复核报告</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf2f7;color:#182a3a;font:15px/1.65 "Microsoft YaHei",sans-serif}
main{max-width:1100px;margin:24px auto;background:white;padding:36px;border-radius:12px}h1{font-size:28px;margin:0}
h2{font-size:20px;margin-top:30px;border-bottom:1px solid #dbe4eb;padding-bottom:8px}h3{font-size:17px}
.muted{color:#526b7b}.intro{background:#f0f6fa;border-left:4px solid #147b95;padding:14px;margin:20px 0}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.metric{padding:14px;background:#f1f6f8;border-radius:8px}
.metric span{display:block;color:#526b7b}.metric strong{display:block;font-size:22px;margin-top:5px}
table{width:100%;border-collapse:collapse;table-layout:fixed}td,th{border:1px solid #dbe4eb;padding:9px;text-align:left;overflow-wrap:anywhere}
th{background:#f0f5f8}figure{margin:18px 0}figcaption{font-weight:bold;margin:8px 0}img{max-width:100%;max-height:640px;object-fit:contain}
.answer,pre{white-space:pre-wrap;overflow-wrap:anywhere}.answer{padding:16px;background:#f7f9fb}pre{font-size:12px;background:#f7f9fb;padding:15px}
li{margin:6px 0}a{color:#086b87}details{margin-top:25px}summary{cursor:pointer}
@media(max-width:650px){main{padding:18px;margin:0;border-radius:0}.metrics{grid-template-columns:repeat(2,1fr)}}
@media print{body{background:white}main{margin:0;padding:0;max-width:none}h2,figcaption{break-after:avoid}figure,.metric{break-inside:avoid}a{color:inherit}details{display:none}}
</style></head><body><main><h1>矿安天眼 · 双期影像复核报告</h1>
<p class="muted">光学影像疑似异常筛查、量测与人工复核辅助 · 可通过浏览器打印保存 PDF</p>
<div class="intro">本报告保存同一次分析的结果。变化量在共同有效像素范围内计算；掩膜占比不是失稳概率，差分区域不是事故数量，掩膜变化不是地面位移。区域及其变化原因仍待人工复核。</div>
<table><tbody><tr><th>报告标识</th><td>''' + _text(packet['report_id']) + '''</td></tr>
<tr><th>生成时间</th><td>''' + _text(packet['generated_at']) + '''</td></tr>''' + times + '''
<tr><th>检测请求标识</th><td>''' + _text(trace.get('request_id')) + '''</td></tr>
<tr><th>分析范围</th><td>共同有效像素 · ''' + _text(record['summary'].get('valid_pixels')) + ''' 像素</td></tr></tbody></table>
<h2>同次变化量测</h2><div class="metrics">''' + cards + '''</div>
<p class="muted">差分连通区域计数与匹配实例计数采用不同口径；匹配区域明细见下方。</p>
<h2>双期影像与变化</h2>''' + ''.join(illustrations) + '''
<h2>自动分析</h2><p class="muted">''' + status + '''</p><div class="answer">''' + _text(record['automatic_analysis']['answer']) + '''</div>
<h2>人工复核交接</h2><p>''' + review_state + '''。复核版本 ''' + _text(record.get('review_revision', 0)) + '''。
人工意见保留原始模型量测，复核人身份尚未独立核验，记录不自动构成事故真值或项目现场试用证明。</p>
<ul>''' + review_rows + '''</ul>
<ul>''' + checks + '''</ul>''' + item_table + '''
<p class="muted">''' + ('区域清单有截断；需回到原始结果核对其余区域。' if record['review_items_truncated'] else '区域清单按本次检测结果交接。') + '''</p>
<h2>知识来源</h2><ul>''' + (''.join(sources) or '<li>本次未引用知识来源</li>') + '''</ul>
<h2>结果来源标识</h2>''' + trace_table + '''
<details><summary>查看请求、权重、输入、配准与量测依据</summary><pre>''' + _json(appendix) + '''</pre></details>
</main></body></html>'''


def export_comparison(result, format='html'):
    packet = comparison_record(result)
    if format == 'html':
        content = comparison_html(result, packet); mime = 'text/html'
    elif format == 'json':
        content = json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        mime = 'application/json'
    else:
        raise ValueError('format 仅支持 html 或 json')
    return {'report_id': packet['report_id'], 'mime_type': mime,
            'filename': '矿安天眼_双期复核报告_' + packet['report_id'] + '.' + format,
            'content_b64': base64.b64encode(content.encode('utf-8')).decode('ascii')}
