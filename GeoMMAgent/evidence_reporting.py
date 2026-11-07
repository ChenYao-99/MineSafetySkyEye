"""Objective evidence reporting baseline, kept separate from geological interpretation."""
import copy,json,math,re

def workflow_status(trace=None, mode='general_dispatch', reason='按任务需要选择执行角色'):
    plan = getattr(trace, 'dispatch_plan', []) if trace else []
    steps = getattr(trace, 'agent_steps', []) if trace else []
    roles = []
    for index, step in enumerate(plan, start=1):
        actual = next((item for item in steps if item.step == index), None)
        roles.append({
            'step': index,
            'name': actual.agent_name if actual else step.get('name', '未执行角色'),
            'status': actual.status if actual else 'not_executed',
        })
    completed = sum(step.status == 'complete' for step in steps)
    return {
        'mode': mode,
        'reason': reason,
        'planned_steps': len(plan),
        'executed_steps': len(steps),
        'completed_steps': completed,
        'interrupted': bool(plan and (len(steps) < len(plan) or completed != len(steps))),
        'roles': roles,
        'single_worker_planned': len(plan) == 1,
        'multiple_workers_executed': len({step.agent_name for step in steps}) > 1,
    }


def objective_facts(det):
    stats=det.get('stats') or {};meta=det.get('band_meta') or {};paired='t1_ratio' in stats
    epochs=[meta.get('t1') or {},meta.get('t2') or {}] if paired else [meta]
    region=(stats.get('regions') or [{}])[0]
    def number(v):return v if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) else None
    def percent(v):
        v=number(v);return None if v is None else round(v*100,6)
    dates=[m.get('acquisition_datetime') for m in epochs]
    facts={'analysis_scope':'common_valid_pixels' if paired else stats.get('scope','not_declared'),
        'kind':'paired_mask_change' if paired else 'single_epoch_mask',
        'reported_regions':None if paired else number(stats.get('num_regions')),
        'area_percent':None if paired or not stats.get('area_ratio_available',True) else percent(stats.get('area_ratio')),
        'largest_area_px':None if paired else number(region.get('area_px',stats.get('max_region_area'))),
        'largest_area_m2':None if paired else number(region.get('area_m2')),
        'largest_perimeter_px':None if paired else number(region.get('perimeter_px')),
        'largest_major_axis_px':None if paired else number(region.get('major_axis_px')),
        'physical_spectral_indices_available':all(bool(m.get('spectral_indices_available')) for m in epochs),
        'weather_quality_provided':all((m.get('observation_quality') or {}).get('source','not_provided')!='not_provided' for m in epochs),
        'capture_times':dates,'unverified_pixels':number(stats.get('unverified_pixels',0)),
        'independent_ground_accuracy_verified':all(bool((m.get('georeference') or {}).get('independent_ground_accuracy_verified')) for m in epochs),
        'accident_confirmed_from_mask':False,'site_stability_confirmed':False,'failure_time_predicted':False,
        'ground_displacement_measured':False,
        'change':{'t1_percent':percent(stats.get('t1_ratio')) if paired else None,'t2_percent':percent(stats.get('t2_ratio')) if paired else None,
                  'net_percentage_points':percent(stats.get('net_ratio')) if paired else None,
                  'added_regions':number(stats.get('n_added_regions')) if paired else None,
                  'expanded_regions':number(stats.get('n_expanded_regions')) if paired else None,
                  'removed_regions':number(stats.get('n_removed_regions')) if paired else None}}
    return facts

def bounded_input(det):
    stats=det.get('stats') or {};meta=det.get('band_meta') or {}
    keys=('scope','num_regions','num_regions_total','area_ratio','area_ratio_available','max_region_area','unverified_pixels','valid_pixels',
          't1_ratio','t2_ratio','net_ratio','n_added_regions','n_expanded_regions','n_removed_regions')
    def minimum_epoch(info):
        geo=info.get('georeference') or {};weather=info.get('observation_quality') or {}
        return {'acquisition_datetime':info.get('acquisition_datetime'),'format':info.get('format'),
                'nir_source':info.get('nir_source'),'spectral_indices_available':bool(info.get('spectral_indices_available')),
                'radiometry_status':info.get('radiometry_status'),
                'observation_quality':{'source':weather.get('source','not_provided')},
                'georeference':{'independent_ground_accuracy_verified':bool(geo.get('independent_ground_accuracy_verified'))}}
    safe_meta={k:minimum_epoch(meta.get(k) or {}) for k in ('t1','t2')} if 't1_ratio' in stats else minimum_epoch(meta)
    scale=det.get('scale_meta') or {};registration=det.get('registration') or {}
    result={'stats':{k:stats[k] for k in keys if k in stats},'first_returned_region':(stats.get('regions') or [{}])[0],
            'band_meta':safe_meta,'scale_meta':{'meters_per_pixel':scale.get('meters_per_pixel'),'source':scale.get('source')},
            'registration':{'status':registration.get('status'),'basis':registration.get('basis'),
                            'independent_ground_accuracy_verified':registration.get('independent_ground_accuracy_verified',False)},
            'model_validation_note':(det.get('model_info') or {}).get('validation_note')}
    result['first_returned_region']={k:v for k,v in result['first_returned_region'].items() if k in ('id','area_px','area_m2','perimeter_px','perimeter_m','major_axis_px','major_axis_m','minor_axis_px')}
    # Preserve declared conditions without exporting geographic transforms or raw pixels.
    roi=det.get('roi') or {};box=roi.get('box_px')
    selected=(stats.get('scope')=='roi' and isinstance(box,list) and len(box)==4 and
              all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in box))
    result['roi_selected']=True if selected else False if stats.get('scope') in ('whole_image','full_image','whole') else None
    result['roi_box_px']=list(box) if selected else None
    result['objective_facts']=objective_facts(det)
    return result

def parse_report(text):
    text=(text or '').strip()
    if text.startswith('```'):text=re.sub(r'^```(?:json)?\s*|\s*```$','',text)
    try:
        value=json.loads(text)
        return value if isinstance(value,dict) else None
    except (ValueError,TypeError):return None

def summary_contract_errors(text,det,allowed_ids):
    """Mechanical consistency checks only; passing is not geological correctness."""
    from report_grounding import (build_grounding,check_structured_geometry,
                                  check_detection_numbers,check_unsupported_safety_claims,check_observation_claims)
    if not isinstance(text,str) or not text.strip():return ['missing_summary']
    stats=det.get('stats') or {};paired='t1_ratio' in stats
    facts={'kind':'compare' if paired else 'single','stats':stats,'band_meta':det.get('band_meta')}
    errors=check_unsupported_safety_claims(text,facts)+check_detection_numbers(text,facts)+check_observation_claims(text,facts)
    grounding=build_grounding(stats.get('regions') or [])
    if grounding:errors.extend(check_structured_geometry(text,grounding)[1])
    errors.extend('unprovided_summary_citation:'+x for x in re.findall(r'\[([A-Z]+-\d{3})\]',text) if x not in allowed_ids)
    if paired:
        net=objective_facts(det)['change']['net_percentage_points']
        for match in re.finditer(r'净变化(?:约|为|[:：]|达|是|\s){0,6}([+-]?\d+(?:\.\d+)?)\s*个百分点',text):
            if net is None or abs(float(match[1])-net)>max(.06,abs(net)*.03):errors.append(match[0])
    return errors

def fixed_evidence_report(original,det,allowed_ids):
    report=copy.deepcopy(original) if isinstance(original,dict) else {}
    from evidence_integrity import bind_structured_report
    previous = bind_structured_report({'facts':report.get('facts'), 'conclusion':report.get('summary')}, objective_facts(det))
    report['facts']=objective_facts(det)
    citations=report.get('citations')
    report['citations']=[x for x in citations if x in allowed_ids] if isinstance(citations,list) else []
    text=report.get('summary')
    if previous['interpretation_status'] == 'withheld_after_fact_correction':
        from report_grounding import fallback_report
        report['raw_generated_summary'] = text
        text = fallback_report({'kind':'compare' if 't1_ratio' in (det.get('stats') or {}) else 'single',
                                'stats':det.get('stats') or {}, 'risk_name':det.get('risk_name','待复核')},
                               '草稿事实已按工具更正，原解释需重新复核')
        report['interpretation_status'] = previous['interpretation_status']
    if not isinstance(text,str):text='仅汇总已提供检测事实；原因与现场状态需人工复核。'
    if summary_contract_errors(text,det,allowed_ids):
        text='模型掩膜不能确认事故或边坡安全；对照原始影像、作业记录和独立资料复核，未知部分保持待核查。'
    report['summary']=text;report['method']='fixed_evidence_fields_and_single_writer'
    return report

def score_objective_report(report,expected,allowed_ids,det=None):
    errors=[];evaluated=0
    def compare(actual,wanted,path):
        nonlocal evaluated
        if isinstance(wanted,dict):
            for key,value in wanted.items():compare(actual.get(key,'__missing__') if isinstance(actual,dict) else '__missing__',value,path+'.'+key)
            return
        evaluated+=1
        if isinstance(wanted,bool):ok=isinstance(actual,bool) and actual==wanted
        elif isinstance(wanted,(int,float)):
            ok=isinstance(actual,(int,float)) and not isinstance(actual,bool) and math.isfinite(actual) and abs(actual-wanted)<=max(.0001,abs(wanted)*1e-5)
        else:ok=actual==wanted
        if not ok:errors.append({'field':path,'expected':wanted,'actual':actual})
    if not isinstance(report,dict):return {'parse_ok':False,'objective_fields':0,'wrong_fields':1,'errors':[{'field':'report','reason':'nonJSONobject'}],'all_objective_fields_correct':False,'unprovided_citations':[]}
    compare(report.get('facts'),expected,'facts')
    cited=report.get('citations',[]);bad=[x for x in cited if x not in allowed_ids] if isinstance(cited,list) else ['invalidcitationformat']
    omitted=not isinstance(report.get('summary'),str) or not report.get('summary','').strip()
    summary_errors=summary_contract_errors(report.get('summary'),det,allowed_ids) if det is not None else ([] if not omitted else ['missing_summary'])
    return {'parse_ok':True,'objective_fields':evaluated,'wrong_fields':len(errors),'errors':errors,
            'all_objective_fields_correct':not errors and not bad,'unprovided_citations':bad,
            'summary_omitted':omitted,'summary_mechanical_contract_errors':summary_errors,
            'report_contract_passed':not errors and not bad and not summary_errors and not omitted,
            'expert_geological_correctness_scored':False}
