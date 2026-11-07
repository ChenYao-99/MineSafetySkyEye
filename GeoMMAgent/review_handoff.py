"""Bounded review tasks and traceable handoff; mask measurements are not field truth."""
import copy
import math
import re
from evidence_reporting import objective_facts

TASK_CONTRACT = {
    'version': 1,
    'purpose': '光学影像疑似异常筛查、量测与人工复核辅助',
    'tasks': [
        {'id': 'INPUT_REVIEW', 'name': '核对影像时间、波段、观测与分析范围'},
        {'id': 'MASK_MEASUREMENT', 'name': '复核预测区域及其量测依据'},
        {'id': 'PAIR_REVIEW', 'name': '在共同有效范围核对前后期变化'},
        {'id': 'REVIEW_HANDOFF', 'name': '交接同次结果、来源和未确认条件'},
    ],
    'can_confirm_accident_or_stability': False,
    'can_predict_failure_time_or_ground_displacement': False,
    'test_record_is_not_user_trial': True,
}


def _number(value, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        return None
    return int(value) if value.is_integer() else value


def _public(value):
    if isinstance(value, dict):
        return {str(k): _public(v) for k, v in value.items()
                if not str(k).startswith('_') and str(k).lower() not in ('api_key', 'authorization', 'image_path', 'overlay_b64', 'image_b64')}
    if isinstance(value, (list, tuple)):
        return [_public(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def _weight_hashes(value):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'weights_sha256' and isinstance(item, str) and re.fullmatch('[a-fA-F0-9]{64}', item):
                found.append(item.lower())
            else:
                found.extend(_weight_hashes(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_weight_hashes(item))
    return sorted(set(found))


def build_review_record(result, max_regions=50):
    if not isinstance(result, dict) or not isinstance(result.get('detection'), dict):
        raise ValueError('复核记录需要同次检测结果')
    det = result['detection']; stats = det.get('stats') or {}
    paired = result.get('region') == 'compare'
    meta = det.get('band_meta') or {}
    epochs = [('t1', meta.get('t1') or {}), ('t2', meta.get('t2') or {})] if paired else [('image', meta)]
    pending = []
    def todo(code, text, epoch=None):
        pending.append({'code': code, 'task': text, 'epoch': epoch, 'status': 'not_reviewed'})
    epoch_evidence = {}
    for epoch, info in epochs:
        geo = info.get('georeference') or {}
        weather = info.get('observation_quality') or {}
        capture = info.get('acquisition_datetime')
        epoch_evidence[epoch] = {
            'acquisition_datetime': capture, 'input_sha256': info.get('input_sha256'),
            'format': info.get('format'), 'nir_source': info.get('nir_source'),
            'radiometry_status': info.get('radiometry_status'),
            'radiometry_source_XML_sha256': info.get('radiometry_source_XML_sha256'),
            'spectral_indices_available': bool(info.get('spectral_indices_available')),
            'georeference': _public(geo), 'observation_quality': _public(weather),
        }
        if not capture:
            todo('ACQUISITION_UNVERIFIED', '核对原始影像的实际采集时间', epoch)
        if weather.get('source', 'not_provided') == 'not_provided':
            todo('WEATHER_UNVERIFIED', '核对云、阴影、雪及可见性；可读取像素不等于天气已核验', epoch)
        elif not weather.get('quality_labels_independently_verified'):
            todo('WEATHER_LABEL_REVIEW_REQUIRED', 'SCL等质量类别尚未独立复核，重要区域应对照原图核查', epoch)
        if not geo:
            todo('GEOGRAPHY_UNVERIFIED', '补充坐标与尺度依据，当前像素量不能直接作为米制量测', epoch)
        elif not geo.get('independent_ground_accuracy_verified'):
            todo('GROUND_LOCATION_UNVERIFIED', '对重要区域核查地面定位；文件地理标签不代表独立测绘精度', epoch)
        if not info.get('spectral_indices_available'):
            todo('PHYSICAL_INDEX_UNAVAILABLE', '缺少可核对光谱或定标条件，当前结果不能提供可靠的物理光谱指数', epoch)
    todo('MODEL_REVIEW_REQUIRED', '对照原始影像、生产作业记录与独立资料复核疑似区域')
    if (det.get('work_context') or {}).get('status')=='matched_public_expected_work_reference':
        todo('WORK_CONTEXT_CAUSE_UNVERIFIED','与公开采矿作业参考重叠的区域仍需核查具体作业日期及变化原因，不能直接确认安全')
    elif (det.get('work_context') or {}).get('status')=='source_reference_unavailable':
        todo('WORK_CONTEXT_REFERENCE_UNAVAILABLE','已登记作业参考尚未通过完整性或格网核查，本次保持原始筛查结果')
    if _number(stats.get('unverified_pixels'), 0):
        todo('CONFLICT_SUPPORT_UNKNOWN', '证据冲突区域保持未知，不能标为正常或安全')
    if paired:
        todo('CHANGE_CAUSE_UNVERIFIED', '区分开采、堆排、复垦、光照与灾害变化；掩膜变化不是地面位移')
    else:
        todo('SINGLE_EPOCH_LIMIT', '单期结果不能独立判断变化原因或事故发展趋势')
    generation = result.get('generation') or {}
    if generation.get('status') == 'fallback':
        todo('AUTOMATIC_ANALYSIS_INCOMPLETE', '本次仅有检测事实汇总；自动分析未完成')
    count = _number(stats.get('num_regions'), 0)
    total = _number(stats.get('num_regions_total'), 0)
    changes = det.get('change_instances') or {}
    change_summary = changes.get('summary') or {}
    current = changes.get('instances') or []
    removed = changes.get('removed_instances') or []
    regions = stats.get('regions') or []
    if paired:
        regions = [(r,'t2') for r in current] + [(r,'t1') for r in removed]
        # Surface change deserves review first. IDs are local to each epoch.
        regions.sort(key=lambda row:(row[0].get('change')=='stable',-(_number(row[0].get('area_px'),0) or 0)))
    fields = ('id', 'bbox', 'centroid', 'area_px', 'area_m2', 'perimeter_px', 'perimeter_m',
              'major_axis_px', 'major_axis_m', 'minor_axis_px', 'minor_axis_m', 'touches_border',
              'work_reference_overlap_pixels','work_reference_overlap_fraction','work_reference_note')
    items = []
    for entry in regions[:max_regions]:
        region,epoch=entry if paired else (entry,'image')
        if not isinstance(region, dict):continue
        item = {k: _public(region[k]) for k in fields if k in region}
        item.update(review_status='not_reviewed', interpretation='模型预测区域，尚未确认为事故边界')
        if paired:
            item.update(review_item_key=epoch+':'+str(region.get('id')),source_epoch=epoch,
                        coordinate_frame='t2_comparison_grid',change_type=region.get('change'),
                        model_mask_area_expansion_ratio=_number(region.get('expansion_ratio'),0),
                        matched_t1_index=region.get('matched_t1_index'))
        items.append(item)
    summary = {'kind': 'paired_mask_change' if paired else 'single_epoch_prediction',
               'scope': stats.get('scope', 'common_valid_support' if paired else 'not_declared'),
               'reported_regions': count, 'reported_total_regions': total,
               'screening_level': det.get('risk_name'), 'screening_level_basis': 'prototype_mask_area_ratio',
               'area_ratio_is_failure_probability': False}
    if paired:
        summary['valid_pixels']=_number(stats.get('valid_pixels'),0)
        summary['mask_difference_components_basis']='pixel difference connected components;not matched instance counts or accident counts'
        summary['matched_instance_summary']=_public(change_summary)
        current_total=sum(_number(change_summary.get(k),0) or 0 for k in ('n_added','n_expanded','n_stable'))
        removed_total=_number(change_summary.get('n_removed'),0) or 0
        source_truncated=bool(change_summary.get('truncated') or change_summary.get('truncated_removed') or
                              current_total>len(current) or removed_total>len(removed))
        for key in ('t1_ratio', 't2_ratio', 'net_ratio'):
            summary[key] = _number(stats.get(key), -1 if key == 'net_ratio' else 0, 1)
        net = summary['net_ratio']
        summary['net_change_percentage_points'] = None if net is None else net * 100
        for key in ('n_added_regions', 'n_expanded_regions', 'n_removed_regions'):
            summary[key] = _number(stats.get(key), 0)
    else:
        source_truncated=False
        summary['area_ratio'] = _number(stats.get('area_ratio'), 0, 1) if stats.get('area_ratio_available', True) else None
        summary['valid_pixels'] = _number(stats.get('valid_pixels', stats.get('total_pixels')), 0)
        summary['unknown_conflict_pixels'] = _number(stats.get('unverified_pixels', 0), 0)
        summary['no_data_pixels'] = _number(stats.get('nodata_pixels'), 0)
    request_id = det.get('request_id'); hashes = _weight_hashes(det.get('model_info') or {})
    trace_complete = bool(request_id and hashes and result.get('report_id'))
    if not trace_complete:
        todo('TRACE_INCOMPLETE', '缺少完整的同次请求或权重标识，需要重新生成可追溯结果')
    return {
        'schema_version': 1, 'task_contract': copy.deepcopy(TASK_CONTRACT),
        'review_state': 'not_reviewed', 'project_field_trial_or_expert_endorsement': False,
        'trace': {'report_id': result.get('report_id'), 'request_id': request_id,
                  'model_tag': det.get('model_tag'), 'threshold': _number(det.get('threshold'), 0, 1),
                  'weights_sha256': hashes, 'trace_complete': trace_complete,
                  'model_validation_status': (det.get('model_info') or {}).get('validation_status')},
        'input_evidence': epoch_evidence, 'observation_support': _public(det.get('input_quality') or {}),
        'selected_roi': _public(det.get('roi')), 'registration': _public(det.get('registration')),
        'work_context': _public(det.get('work_context')),
        'measurement_scale': _public(det.get('scale_meta') or {}),
        'summary': summary, 'review_items': items,
        'objective_facts': objective_facts(det), 'objective_facts_source': 'same_detector_result',
        'listed_regions': len(items), 'review_items_truncated': bool(source_truncated or len(regions) > len(items) or (total is not None and total > len(items))),
        'pending_checks': pending, 'knowledge_sources': _public(result.get('knowledge') or []),
        'automatic_analysis': {'generation': _public(generation), 'quality': _public(result.get('quality') or {}),
                               'answer': result.get('final_answer') or ''},
        'accepted_human_review_decisions': [],
    }
