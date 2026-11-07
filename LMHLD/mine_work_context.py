"""Exact-input public work evidence; adds competing-cause cues, never edits detections."""
from pathlib import Path
import json
import cv2,numpy as np,rasterio
from model_registry import file_sha256
ROOT=Path(__file__).resolve().parent
REGISTRY=ROOT/'datasets/slats_operation_development_20261005/runtime_work_context_registry.json'

def annotate_known_work(mask,band_meta,scope_valid,stats,registry_path=REGISTRY):
    if not registry_path.exists():return None
    try:
        entries=json.loads(registry_path.read_text(encoding='utf-8'))['entries']
        if not isinstance(entries,dict):raise ValueError('Invalid work reference registry')
        entry=entries.get(band_meta.get('input_sha256'))
        if entry is None:return None
        ref_path=Path(entry['reference']);mask_path=Path(entry['mask'])
        if not ref_path.resolve().is_relative_to((ROOT/'datasets').resolve()) or not mask_path.resolve().is_relative_to((ROOT/'datasets').resolve()):raise ValueError('Reference path outside dataset root')
        if file_sha256(ref_path)!=entry['reference_sha256'] or file_sha256(mask_path)!=entry['mask_sha256']:raise ValueError('Registered work evidence changed')
        reference=json.loads(ref_path.read_text(encoding='utf-8'))
        with rasterio.open(mask_path) as src:core=src.read(1)>0
        if core.shape!=mask.shape or np.asarray(scope_valid).shape!=mask.shape:raise ValueError('Context observation grid mismatch')
        if band_meta.get('acquisition_datetime')!=reference['actual_dates']['post']:raise ValueError('Known source acquisition mismatch')
        observed=np.asarray(scope_valid,bool);scoped=(np.asarray(mask)>0)&observed
        core=core&observed
        _,labels,component_stats,_=cv2.connectedComponentsWithStats(scoped.astype(np.uint8),connectivity=8)
        # Match the upload route's min_region_area=10. The original mask is kept.
        eligible=component_stats[:,cv2.CC_STAT_AREA]>=10;eligible[0]=False
        reported=eligible[labels]
        foreground=int(reported.sum());overlap=int((reported&core).sum())
        components={};ambiguous=set()
        for component_id,values in enumerate(component_stats[1:],start=1):
            if not eligible[component_id]:continue
            key=tuple(int(v) for v in values)
            if key in components:ambiguous.add(key)
            else:components[key]=component_id
        notes=[]
        for region in stats.get('regions') or []:
            region_id=region.get('id')
            if not isinstance(region_id,int):continue
            bbox=region.get('bbox') or [];area=region.get('area_px')
            if len(bbox)!=4 or not isinstance(area,int):continue
            key=tuple(bbox)+(area,)
            if key in ambiguous or key not in components:continue
            selected=labels==components[key]
            count=int((selected&core).sum())
            if count:
                ratio=count/area
                note='与公开年度采矿清表及现期作业外观参考重叠，原因待人工复核'
                region.update(work_reference_overlap_pixels=count,work_reference_overlap_fraction=round(ratio,6),work_reference_note=note)
                notes.append({'instance_id':region_id,'overlap_pixels':count,'fraction':round(ratio,6)})
        return {'status':'matched_public_expected_work_reference','input_match':'exact_original_or_verified_SCL_packet_SHA256',
                'source_period':'2023年8月至2024年8月（公开资料的名义年度范围）',
                'source_period_raw':reference['source_nominal_period'],'reference_confidence':reference['confidence'],
                'source_title':'昆士兰州政府SLATS年度采矿清表分类',
                'source_url':entry['source_url'],'source_geojson_sha256':reference['source_sha256'],
                'reference_sha256':entry['reference_sha256'],'work_reference_pixels_in_scope':int(core.sum()),
                'overlap_basis':'same reported connected components with minimum10pixels;no discarded-noise component counted',
                'predicted_pixels_overlapping_work_reference':overlap,
                'fraction_of_predicted_pixels_overlapping_reference':round(overlap/foreground,6) if foreground else None,
                'same_request_region_overlaps':notes,'raw_prediction_pixels_weights_and_screening_level_unchanged':True,
                'exact_daily_change_cause_verified':False,'geotechnical_safety_or_accident_absence_verified':False,
                'note':'重叠部分具有公开采矿作业参考，应结合生产与现场记录复核变化原因；不能直接确认事故或安全。'}
    except (OSError,ValueError,KeyError) as exc:
        return {'status':'source_reference_unavailable','raw_prediction_pixels_weights_and_screening_level_unchanged':True,
                'note':'已登记作业参考未通过完整性或格网检查，本次保持原始筛查结果。'}
