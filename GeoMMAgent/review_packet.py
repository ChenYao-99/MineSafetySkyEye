"""Readonly offline handoff from one archived record; no inference or invented geometry."""
import copy,html,json,math
from comparison_reporting import _image,_public


def text(value):return html.escape('未提供' if value is None else str(value),quote=True)


def review_html(result,independent_evidence=None):
    record=_public(copy.deepcopy(result['review_handoff']));det=result['detection'];paired=result.get('region')=='compare'
    facts=result.get('objective_facts') or record['objective_facts'];trace=record['trace']
    if not trace.get('trace_complete'):raise ValueError('缺少完整的同次来源，不能导出复核报告')
    raw=_image(det.get('t2_rgb_b64') if paired else det.get('rgb_b64'))
    overlay=_image(det.get('t2_overlay_b64') if paired else det.get('overlay_b64'))
    shape=det.get('size');geometry_ok=isinstance(shape,list) and len(shape)==2 and all(isinstance(v,int) and not isinstance(v,bool) and v>1 for v in shape)
    markers=record.get('manual_annotations') or [];boxes=[]
    if geometry_ok:
        h,w=shape
        for index,row in enumerate(markers):
            annotation=row.get('manual_annotation') or {};box=annotation.get('box_px')
            if not isinstance(box,list) or len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box):raise ValueError('补记位置的像素坐标无效')
            x0,y0,x1,y1=box
            if not (0<=x0<x1<=w and 0<=y0<y1<=h) or annotation.get('source',{}).get('report_id')!=result['report_id']:
                raise ValueError('补记位置与本次影像范围或报告标识不一致')
            boxes.append(f'<g class="marker"><rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}"/><text x="{x0+3}" y="{max(12,y0-4)}">补记{index+1}</text></g>')
    if raw and geometry_ok:
        picture=f'<svg id="scene" viewBox="0 0 {w} {h}" role="img" aria-label="同次原图显示与人工补记位置"><image id="raw" width="{w}" height="{h}" href="{raw}"/>'
        if overlay:picture+=f'<image id="prediction" width="{w}" height="{h}" href="{overlay}" style="display:none"/>'
        picture+='<g id="markers">'+''.join(boxes)+'</g></svg>'
        picture+='<div class="controls"><button data-view="raw">原图显示</button><button data-view="prediction"'+('' if overlay else ' disabled')+'>候选叠加</button><label><input id="showMarkers" type="checkbox" checked> 显示人工补记</label></div>'
    elif raw:picture='<img class="scene" src="'+raw+'" alt="同次原图显示"><p>原图尺寸未保存，未叠加定位框。</p>'
    else:picture='<p class="notice">本记录未保存原图显示，请另行核对原始影像。模型叠加图不替代原图。</p>'+('<img class="scene" src="'+overlay+'" alt="模型叠加显示">' if overlay else '')
    metrics=[('T1候选覆盖占比',facts['change'].get('t1_percent'),'%'),('T2候选覆盖占比',facts['change'].get('t2_percent'),'%'),('净变化',facts['change'].get('net_percentage_points'),'个百分点')] if paired else [('模型候选覆盖占比',facts.get('area_percent'),'%'),('模型实例数',facts.get('reported_regions'),'处')]
    cards=''.join('<div><span>'+text(name)+'</span><strong>'+text(value)+(' '+unit if value is not None else '')+'</strong></div>' for name,value,unit in metrics)
    cards+='<div><span>独立保存的人工定位补记</span><strong>'+str(len(markers))+' 处</strong></div>'
    entries=[]
    for name,meta in record['input_evidence'].items():
        entries.extend([(name+'采集时间',meta.get('acquisition_datetime')),(name+'原始输入SHA-256',meta.get('input_sha256'))])
    entries.extend([('检测请求',trace.get('request_id')),('模型权重SHA-256','\n'.join(trace.get('weights_sha256') or [])),('阈值',trace.get('threshold')),('分析范围',facts.get('analysis_scope')),('复核版本',record.get('review_revision',0)),('机器快照SHA-256',(result.get('archive') or {}).get('snapshot_sha256'))])
    provenance='<table>'+''.join('<tr><th>'+text(name)+'</th><td>'+text(value)+'</td></tr>' for name,value in entries)+'</table>'
    rows=[]
    for row in record.get('review_history',[]):
        box=(row.get('manual_annotation') or {}).get('box_px')
        where='人工定位框 '+text(box) if box else text(row.get('target_key'))
        rows.append('<article class="review"><b>版本 '+text(row.get('revision'))+' · '+text(row.get('status_label'))+'</b><p>'+where+'</p><p>'+text(row.get('note'))+'</p><p class="muted">依据：'+text(row.get('evidence'))+'</p><p class="muted">'+text(row.get('reviewer'))+' · '+text(row.get('created_at'))+'</p></article>')
    checks='<ul>'+''.join('<li>'+text(item.get('task'))+'</li>' for item in record.get('pending_checks',[]))+'</ul>'
    performance=''
    if independent_evidence and independent_evidence.get('weights_sha256') in trace.get('weights_sha256',[]):
        score=independent_evidence['metrics']
        performance='<section class="notice"><b>该权重已有的独立研究验收：未达标</b><p>CL0002公开清单可评分范围内的单期对照：召回率 '+text(round(score['recall']*100,2))+'%，精确率 '+text(round(score['precision']*100,2))+'%，IoU '+text(round(score['IoU']*100,2))+'%。这些是另一个矿区的结果，不是本页案例的精度，也不能外推为现场误报率。</p><p>候选叠加不能作为完整异常排查；需对照原图核查未被标出的区域。</p></section>'
    appendix=text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False))
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>矿安天眼 · 离线复核交接</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef3f8;color:#20354a;font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1150px;margin:28px auto;background:white;padding:32px;border-radius:16px}h1{font-size:27px;margin:0}h2{font-size:20px;margin-top:30px}.muted{color:#60758a}.notice{background:#fff7e8;border-left:4px solid #dda535;padding:16px;border-radius:8px;margin:18px 0}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.metrics>div{background:#edf5fc;padding:16px;border-radius:10px}.metrics span{display:block;color:#526d86}.metrics strong{font-size:23px}.scene,#scene{display:block;width:100%;max-height:720px;object-fit:contain;background:#f5f7fa;border-radius:12px}.marker rect{fill:none;stroke:#ffae32;stroke-width:2;vector-effect:non-scaling-stroke}.marker text{fill:#102e49;paint-order:stroke;stroke:#ffe3aa;stroke-width:3;font-size:14px}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:12px 0}button{background:#12507f;color:white;border:0;border-radius:8px;padding:10px 16px;font:inherit;cursor:pointer}button:disabled{opacity:.4}table{width:100%;border-collapse:collapse;table-layout:fixed}td,th{padding:10px;border:1px solid #dbe4ed;text-align:left;overflow-wrap:anywhere}th{width:210px;background:#f3f7fa}.review{border:1px solid #dbe4ed;border-radius:10px;padding:16px;margin:12px 0}.review p{white-space:pre-wrap;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f8fc;padding:16px}summary{cursor:pointer;font-weight:bold}@media(max-width:700px){main{margin:0;padding:20px;border-radius:0}.metrics{grid-template-columns:1fr}th{width:125px}}@media print{body{background:white}main{margin:0}.controls{display:none}section,article{break-inside:avoid}}
</style></head><body><main><h1>矿安天眼 · 离线复核交接</h1><p class="muted">报告 '''+text(result['report_id'])+' · '+text(result.get('generated_at'))+'''</p>
<div class="notice">本文件读取同一次归档，不重新检测或调用云端。橙色框是人工核查位置，不是事故边界；框面积不计入模型量测。复核身份与意见未经独立核验，记录不证明现场试用、事故原因或地面稳定性。</div>
<div class="metrics">'''+cards+'</div>'+performance+'<h2>'+('T2原图与核查位置' if paired else '原图与核查位置')+'</h2>'+picture+'<h2>来源与同次量测</h2>'+provenance+'<h2>复核意见与修改历史</h2>'+(''.join(rows) if rows else '<p>尚未记录复核意见。</p>')+'<h2>仍需核查的条件</h2>'+checks+'<details><summary>完整结构化复核记录</summary><pre>'+appendix+'</pre></details>'+'''
</main><script>document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{const layer=document.getElementById('prediction');if(layer)layer.style.display=button.dataset.view==='prediction'?'':'none';}));const toggle=document.getElementById('showMarkers');if(toggle)toggle.addEventListener('change',()=>{document.getElementById('markers').style.display=toggle.checked?'':'none';});</script></body></html>'''
