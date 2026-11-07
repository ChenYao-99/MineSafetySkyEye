"""Structured numeric provenance for generated reports (standard library only)."""
import math
import re
import unicodedata


METRICS = {
    "area": ("area_px", "area_m2"),
    "perimeter": ("perimeter_px", "perimeter_m"),
    "major_axis": ("major_axis_px", "major_axis_m"),
    "minor_axis": ("minor_axis_px", "minor_axis_m"),
}


def wrap_report_text(text, width=100):
    """Wrap CJK by display width before rendering (matplotlib tight bbox cannot do this)."""
    output = []
    for original in text.split("\n"):
        line, cells = "", 0
        for char in original:
            size = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
            if cells + size > width and line:
                output.append(line)
                line, cells = "", 0
            line += char
            cells += size
        output.append(line)
    return "\n".join(output)
GEOMETRY = re.compile(
    r"(?P<metric>实例面积|区域面积|面积(?!占比|比例|率)|周长|长轴|短轴|长度|宽度)"
    r"[^\d\n;；。]{0,12}(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>平方米|平方像素|m²|m2|px²|px2|像素|px|米|m)(?![A-Za-z])", re.I)
NAMES = {"实例面积": "area", "区域面积": "area", "面积": "area", "周长": "perimeter",
         "长轴": "major_axis", "长度": "major_axis", "短轴": "minor_axis", "宽度": "minor_axis"}


def build_grounding(regions, n_total=None, meters_per_pixel=None):
    facts, numbers = [], set()
    if n_total is not None:
        numbers.add(str(n_total))
    for region in regions:
        for fields in METRICS.values():
            for field in fields:
                value = region.get(field)
                if value is None and meters_per_pixel and field.endswith(("_m", "_m2")):
                    px = region.get(field.replace("_m2", "_px").replace("_m", "_px"))
                    if px is not None:
                        value = float(px) * float(meters_per_pixel) ** (2 if field.endswith("_m2") else 1)
                if value is not None and math.isfinite(float(value)):
                    facts.append({"instance_id": region.get("id"), "field": field, "value": float(value)})
                    numbers.add(str(value))
    return {"numbers": sorted(numbers), "measurements": facts} if facts else None


def check_structured_geometry(answer, grounding):
    facts = grounding.get("measurements", [])
    bad, cited = [], False
    for match in GEOMETRY.finditer(answer or ""):
        metric = NAMES[match["metric"]]
        unit = match["unit"].lower()
        metric_px, metric_m = METRICS[metric]
        field = metric_px if "像素" in unit or unit.startswith("px") else metric_m
        # Reject a length in square units, or area in linear metres.
        square = unit in ("平方米", "平方像素", "m²", "m2", "px²", "px2")
        wrong_dimension = (metric != "area" and square) or (metric == "area" and unit in ("米", "m"))
        prefix = (answer or "")[max(0, match.start() - 90):match.start()]
        ids = re.findall(r"(?:实例\s*#?|#)\s*(\d+)", prefix)
        candidates = [f for f in facts if f["field"] == field
                      and (not ids or str(f["instance_id"]) == ids[-1])]
        value = float(match["value"])
        matched = not wrong_dimension and any(
            abs(value - f["value"]) <= max(.51 if field.endswith("_px") else .05, abs(f["value"]) * .03)
            for f in candidates)
        if matched:
            cited = True
        else:
            bad.append(match.group(0))
    return cited, bad[:8]


def check_detection_numbers(answer, facts):
    """Verify explicit coverage/count claims against the selected detection scope."""
    if not facts:
        return []
    stats, bad = facts.get("stats", {}), []
    text = answer or ''
    def consistent(raw, expected):
        if expected is None:
            return False
        precision = len(raw.split('.')[1]) if '.' in raw else 0
        return abs(float(raw) - float(expected)) <= .5 * 10 ** (-precision) + 1e-6
    if facts.get('kind') == 'compare' or 't1_ratio' in stats:
        for match in re.finditer(r'(T[12])[^\d\n。；;]{0,24}([+-]?\d+(?:\.\d+)?)\s*%', text, re.I):
            key = 't1_ratio' if match[1].upper() == 'T1' else 't2_ratio'
            if not consistent(match[2], None if stats.get(key) is None else stats[key]*100):
                bad.append(match[0])
        for match in re.finditer(r'(?:净变化|净增|净减|净增加|净减少)(?:约|为|[:：]|达|是|\s){0,6}([+-]?\d+(?:\.\d+)?)\s*(个百分点|%)', text):
            if match[2] != '个百分点' or not consistent(match[1], None if stats.get('net_ratio') is None else stats['net_ratio']*100):
                bad.append(match[0])
        # Rounded displays may be valid individually but are not strict equalities.
        for match in re.finditer(r'([+-]?\d+(?:\.\d+)?)\s*个百分点\s*[（(]?\s*(?:即|等于|也就是)\s*([+-]?\d+(?:\.\d+)?)\s*个百分点',text):
            suffix=text[match.end():match.end()+35]
            if abs(float(match[1])-float(match[2]))>1e-9 and not re.search(r'四舍五入|保留[一二三\d]+位|舍入',suffix):
                bad.append('unequal_percentage_points:'+match[0])
        for label, key in [('新增','n_added_regions'),('扩大','n_expanded_regions'),('消退','n_removed_regions'),('消失','n_removed_regions')]:
            for match in re.finditer(label+r'(?:(?!新增|扩大|消退|消失|[与和及、])[^\d\n。；;]){0,12}(\d+)\s*处', text):
                if stats.get(key) is None or int(match[1]) != stats[key]:
                    bad.append(match[0])
        return bad
    for match in re.finditer(r"(?:异常|风险|滑坡)?(?:掩膜|区域)?面积占比(?:约|为|[:：]|达|是|\s){0,6}"
                             r"(\d+(?:\.\d+)?)\s*%", answer or ""):
        # A reference rule listing threshold boundaries is not this image's ratio.
        prefix=text[max(0,match.start()-12):match.start()]
        suffix=text[match.end():match.end()+50]
        threshold_list=(re.search(r'以|阈值|分界|按',prefix) and
                        re.match(r'\s*(?:[、,，]|[–—~至到-])\s*\d+(?:\.\d+)?\s*%',suffix) and
                        re.search(r'阈值|分界|分级|划为|色',suffix))
        if threshold_list:continue
        if stats.get('area_ratio_available',True) is False:
            bad.append(match[0])
            continue
        expected = None if stats.get('area_ratio') is None else float(stats['area_ratio']) * 100
        if not consistent(match[1], expected):
            bad.append(match[0])
    for match in re.finditer(r"(?:共|检测到|输出|识别到)\s*(\d+)\s*处", answer or ""):
        if int(match[1]) != int(stats.get("num_regions", 0)):
            bad.append(match[0])
    return bad


def check_report_citations(answer, question):
    allowed = set(re.findall(r"\[([A-Z]+-\d{3})\]", question or ""))
    cited = set(re.findall(r"\[([A-Z]+-\d{3})\]", answer or ""))
    return not allowed or bool(cited) and cited <= allowed


def check_source_detail_claims(answer, question):
    """Do not invent standards or chapter contents absent from provided references."""
    def normalize(raw):return re.sub(r'\s+','',raw).upper()
    text=answer or '';source=question or '';bad=[]
    standard=r'GB(?:/T)?\s*\d{3,6}(?:[-—]\d{4})?'
    known={normalize(x) for x in re.findall(standard,source,re.I)}
    for match in re.finditer(standard,text,re.I):
        if normalize(match[0]) not in known:bad.append('unprovided_standard:'+match[0])
    known_chapters=set(re.findall(r'第[一二三四五六七八九十百\d]+章',source))
    for match in re.finditer(r'第[一二三四五六七八九十百\d]+章',text):
        if match[0] not in known_chapters:bad.append('unprovided_chapter:'+match[0])
    return bad[:8]


def report_input_context(facts):
    """Minimal authoritative observation context for both writer and repair calls."""
    import json
    from evidence_reporting import bounded_input
    facts = facts or {}
    meta = facts.get('band_meta') or {}
    epochs = {k:meta.get(k) or {} for k in ('t1','t2')} if facts.get('kind')=='compare' else {'image':meta}
    values = {name:{'acquisition_datetime':item.get('acquisition_datetime'),
                    'nir_source':item.get('nir_source'),
                    'spectral_indices_available':item.get('spectral_indices_available') is True,
                    'observation_quality_source':(item.get('observation_quality') or {}).get('source','not_provided')}
              for name,item in epochs.items()}
    declared=bounded_input(facts)
    return ('同次请求的输入依据（未提供保持未知，参考资料不能补成现场量测）：\n' +
            json.dumps({'case_id':facts.get('case_id'),'epochs':values,
                        'analysis_scope':facts.get('stats',{}).get('scope','common_valid_pixels' if facts.get('kind')=='compare' else 'not_declared'),
                        'roi_selected':declared['roi_selected'],'roi_box_px':declared['roi_box_px'],
                        'objective_facts':declared['objective_facts'],
                        'scope_rule':'ROI是否已圈定按roi_selected和同次像素框判断；与现场工程分区是否一致仍需复核。已提供日期、实测波段和定标状态不得重复称为未提供；这些声明不代表独立地面精度已验证。',
                        'ground_displacement_measured':False,'failure_time_predicted':False},
                       ensure_ascii=False,separators=(',',':')))


def check_known_input_claims(answer,facts):
    """Reject explicit denial/uncertainty about conditions already declared by tools.

    Selection of a pixel ROI is separate from its suitability for an engineering
    zone; measured/calibrated bands are separate from independent field accuracy.
    """
    from evidence_reporting import bounded_input
    if not facts:return []
    text=answer or '';declared=bounded_input(facts);bad=[]
    if declared['roi_selected'] is not None:
        for match in re.finditer(r'(?:ROI|监测范围|分析范围)[^。；;\n]{0,18}(?:未(?:在输入中)?(?:明确)?圈定|尚未圈定|没有圈定|未(?:明确)?设定|是否(?:已经|已)?圈定)',text,re.I):
            # A statement about a future or hypothetical ROI is not about this task.
            clause=text[max(0,match.start()-12):match.end()]
            if re.search(r'后续|如果|若|假设|未圈定时',clause):continue
            if declared['roi_selected'] or '是否' in match[0]:bad.append('known_ROI_selection_repeated_as_unknown:'+match[0])
    meta=facts.get('band_meta') or {};paired='t1_ratio' in (facts.get('stats') or {})
    epochs={k.upper():meta.get(k) or {} for k in ('t1','t2')} if paired else {'IMAGE':meta}
    for match in re.finditer(r'(?:未提供|缺少|没有|尚未提供)(?:可用的|有效的|任何|真实|实测|近红外|\s){0,6}(?:NIR|近红外)(?:波段)?',text,re.I):
        suffix=text[match.end():match.end()+10]
        if re.match(r'(?:细节|详情|校验|验证|独立|地面|原始)',suffix):continue
        before=re.split(r'[。；;\n]',text[:match.start()])[-1]
        labels=re.findall(r'T[12]',before,re.I)
        items=[epochs[labels[-1].upper()]] if paired and labels else list(epochs.values())
        if items and all(v.get('nir_source')=='measured' and v.get('spectral_indices_available') is True for v in items):
            bad.append('provided_measured_NIR_denied:'+match[0])
    for match in re.finditer(r'(?:未说明|未提供|缺少)[^。；;\n]{0,16}(?:大气校正|辐射定标)(?:状态|信息)?',text):
        if re.search(r'独立|地面|现场',match[0]):continue
        items=list(epochs.values())
        if items and all(v.get('radiometry_status')=='verified_local_product_XML' for v in items):
            bad.append('provided_product_radiometry_status_denied:'+match[0])
    return bad[:8]


def check_observation_claims(answer, facts):
    """Check explicit case, capture-date and unsupported measurement claims.

    This covers declared claim forms, not every paraphrase or geological meaning.
    Date checks distinguish capture dates from dates in reference citations.
    """
    from datetime import date, datetime
    if not facts:
        return []
    text = answer or ''; bad=[]
    meta = facts.get('band_meta') or {}
    paired = facts.get('kind')=='compare' or 't1_ratio' in facts.get('stats',{})
    epochs = {k.upper():meta.get(k) or {} for k in ('t1','t2')} if paired else {'IMAGE':meta}
    negations = r'不能|无法|不足以|不代表|不等于|不得|并非|未证实|尚未|不可|未提供|没有|不应|未测得|未实测|无法确认'
    def negated(start):
        prefix = re.split(r'[。；;\n]',text[:start])[-1][-24:]
        return bool(re.search(negations,prefix))
    def normal_date(raw):
        if not isinstance(raw,str):return None
        try:
            return datetime.fromisoformat(raw.replace('Z','+00:00')).date().isoformat()
        except ValueError:
            try:return date.fromisoformat(raw[:10]).isoformat()
            except ValueError:return None
    date_pattern = r'(\d{4})\s*(?:-|/|年)\s*(\d{1,2})\s*(?:-|/|月)\s*(\d{1,2})(?:日)?'
    claim_pattern = (r'(?:(T[12])\s*)?(?:(?:本期|本次|当前|前期|后期)\s*)?'
                     r'(?:卫星|遥感)?(?:影像|图像)?\s*(?:的)?(?:采集|获取|拍摄|成像)'
                     r'(?:时间|日期|时相|于)?\s*(?:为|是|[:：]|在)?\s*'+date_pattern)
    for match in re.finditer(claim_pattern,text,re.I):
        if negated(match.start()):continue
        try:claimed=date(int(match[2]),int(match[3]),int(match[4])).isoformat()
        except ValueError:
            bad.append('invalid_capture_date:'+match[0]);continue
        label=(match[1] or '').upper()
        if not label and paired:
            prefix=re.split(r'[。；;\n，,]',text[:match.start()])[-1][-16:]
            previous=re.findall(r'T[12]',prefix,re.I)
            if previous:label=previous[-1].upper()
        expected=[normal_date(epochs[label].get('acquisition_datetime'))] if label in epochs else [normal_date(v.get('acquisition_datetime')) for v in epochs.values()]
        if claimed not in expected:bad.append('unprovided_or_wrong_capture_date:'+match[0])
    for match in re.finditer(r'(?:案例(?:编号|标识|ID)|case[_ ]?id)\s*(?:\*\*)?\s*(?:为|是|[:：=])\s*(?:\*\*)?\s*["\']?([A-Za-z0-9_-]{8,64})',text,re.I):
        if not negated(match.start()) and str(match[1]) != str(facts.get('case_id')):
            bad.append('foreign_case_identifier:'+match[0])
    if facts.get('stats',{}).get('scope') in ('whole_image','full_image'):
        for match in re.finditer(r'(?:本次|当前|均限定|所有面积占比|掩膜总面积)[^。；;\n]{0,32}(?:ROI范围|ROI区域|圈定监测范围)',text,re.I):
            if not negated(match.start()) and not re.search(r'若|如果|后续|假设|不得|不能|不等于',match[0]):
                bad.append('wrong_analysis_scope:'+match[0])
    for match in re.finditer(r'NDVI\s*(?:指数|值)?\s*(?:约为|为|是|[:：=]|达到|约)?\s*[-+]?\d+(?:\.\d+)?',text,re.I):
        if negated(match.start()):continue
        prefix=re.split(r'[。；;\n]',text[:match.start()])[-1]
        labels=re.findall(r'T[12]',prefix,re.I)
        items=[epochs[labels[-1].upper()]] if paired and labels else list(epochs.values())
        if not all(v.get('spectral_indices_available') is True for v in items):
            bad.append('spectral_measurement_unavailable:'+match[0])
    patterns = [
        ('unsupported_failure_forecast',r'(?:预计|预测|预判|将会|将在)[^。；;\n]{0,28}(?:发生滑坡|发生失稳|边坡失稳|发生崩塌)'),
        ('unsupported_warning_lead_time',r'(?:本系统|本项目|系统)[^。；;\n]{0,12}(?:提前|提前量)[^。；;\n]{0,8}\d+(?:\.\d+)?\s*(?:天|小时)'),
        ('unsupported_ground_displacement',r'(?:现场监测显示|现场实测|实测|测得)[^。；;\n]{0,20}(?:位移|裂缝宽度)[^。；;\n]{0,8}\d+(?:\.\d+)?\s*(?:毫米|厘米|米|mm|cm|m)'),
        ('no_evidence_is_not_a_downgrade_rule',r'未(?:获取|收到|发现|取得)[^。；;\n]{0,35}(?:自动|直接)(?:降级|解除预警|排除风险)'),
        ('mask_area_level_is_not_model_confidence',r'(?:原型筛查分级|筛查分级|筛查等级)[^。；;\n]{0,25}(?:为|表示|代表|反映)(?:模型)?置信度'),
    ]
    if facts.get('stats',{}).get('unverified_pixels',0):
        patterns.append(('conflict_marked_as_background',r'(?:冲突|未判定|未知)[^。；;\n]{0,32}(?:已归为|已计为|确认为|判为|归入|计入)(?:正常|安全|背景)'))
    for kind, pattern in patterns:
        for match in re.finditer(pattern,text,re.I):
            if not negated(match.start()) and not re.search(negations,match[0]):
                bad.append(kind+':'+match[0])
    return bad[:12]


def check_unsupported_safety_claims(answer, facts):
    """Screening masks do not substantiate affirmative geotechnical safety/forecast claims."""
    if not facts:
        return []
    unsupported=[]
    pattern=r'已排除(?:滑坡|失稳|事故|隐患)(?:风险)?|(?:边坡|矿区)(?:处于|已|是|为)?(?:安全|稳定)(?:状态)?|不会(?:发生)?(?:滑坡|失稳)|无(?:滑坡|失稳)风险'
    for match in re.finditer(pattern,answer or ''):
        suffix=(answer or '')[match.end():match.end()+12]
        if re.match(r'(?:性|监测|管理|评估|评价|研究|设计|制度|分析|要求|防护|保障|技术|措施|需|待|尚未|未)',suffix):
            continue
        prefix=re.split(r'[。；;\n]',(answer or '')[:match.start()])[-1][-20:]
        if re.search(r'不能|无法|不足以|不代表|不等于|不得|并非|未证实|尚未|复核|核实|评估|检查',prefix):
            continue
        unsupported.append(match.group(0))
    return unsupported[:8]


def fallback_report(facts, reason):
    """A deterministic summary of measured detection output; never fabricates an agent result."""
    facts = facts or {}
    stats = facts.get("stats", {})
    level = facts.get("risk_name", "待复核")
    if facts.get("kind") == "compare":
        return (f"自动评估未完成：{reason}。以下为两期检测事实汇总。\n"
                f"共同有效范围内，T1异常掩膜面积占比{stats.get('t1_ratio', 0)*100:.2f}%，"
                f"T2占比{stats.get('t2_ratio', 0)*100:.2f}%，"
                f"净变化{stats.get('net_ratio', 0)*100:+.2f}个百分点。"
                f"新增{stats.get('n_added_regions', 0)}处，扩大{stats.get('n_expanded_regions', 0)}处，"
                f"消退{stats.get('n_removed_regions', 0)}处；T2原型筛查分级为{level}。\n"
                "上述变化来自影像与模型掩膜，不是现场变形测量；配准、光照和生产活动均可能影响差异。"
                "监测建议：专业人员结合两期采集时间、范围、分辨率、配准状态与现场记录复核，"
                "保留疑似变化区及其来源。影像中的新增或消退不能独立证明滑坡发生或边坡变得安全。"
                "本次未获得有效的多智能体分析，后续处置按现场制度与专业人员研判执行。")
    scope = "圈定监测范围" if stats.get("scope") == "roi" else "本次影像范围"
    ratio = float(stats.get("area_ratio", 0)) * 100
    count = stats.get("num_regions", 0)
    coverage = f'面积占比{ratio:.1f}%，' if stats.get('area_ratio_available',True) else '没有可判定像素，面积占比不可计算，'
    if stats.get('unverified_pixels',0):scope='有效非冲突像素范围'
    report = (f"自动评估未完成：{reason}。以下为检测工具输出的事实汇总。\n"
              f"分析范围：{scope}。疑似异常区域共{count}处，" + coverage +
              f"原型筛查分级为{level}。该分级由预测掩膜面积比例计算，"
              "不代表边坡失稳概率、法定预警等级或已确认的滑坡数量。\n")
    if stats.get('unverified_pixels',0):
        report += (f"另有{stats['unverified_pixels']}像素存在证据冲突，尚未判定；"
                   "未判定部分不计为正常背景，本次结果须人工复核。\n")
    regions = stats.get("regions") or []
    if regions:
        r = regions[0]
        report += (f"最大返回实例面积{r.get('area_px', stats.get('max_region_area', 0))}像素，"
                   f"周长{r.get('perimeter_px', 0)}像素，长轴{r.get('major_axis_px', 0)}像素。\n")
    report += ("监测建议：由矿山安全或地质人员复核疑似区域，结合现场巡查、"
               "已有变形监测记录及经批准的监测方案研判；当前结果不足以独立确认事故隐患。"
               "影像的采集时间、地面分辨率、云雪遮挡与模型适用区域应一并核查。"
               "工程处置与人员行动应依现场制度和专业人员研判执行。"
               "本次未获得有效的多智能体分析，请保留检测统计与来源，待服务恢复后再生成分析报告。")
    return report
