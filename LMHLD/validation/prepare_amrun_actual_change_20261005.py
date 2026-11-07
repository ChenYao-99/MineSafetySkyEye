"""Register a dated actual-work difficult example before image review or model use."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, copy, json, sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
OUT=ROOT/'datasets/source_backed_operation_change_20261005'
EVIDENCE=ROOT/'datasets/actual_operation_evidence_20261005/Amrun'

def run(full_window=False):
    old=json.loads((ROOT/'datasets/diverse_operation_pairs_20261004/site_plan.json').read_text(encoding='utf-8'))
    site=copy.deepcopy(old['sites']['Amrun'])
    site.update(site_id='AU_AMRUN202309',mine_group='Amrun',role='auxiliary_development_train',
                not_a_new_blind_mine=True, normal_or_stability_labels_available=False,
                scope='actual-work context and visible-change review; no field-verified pixel safety labels')
    # The audited hectare total covers the whole operation. Downloaded imagery is
    # a pre-existing subwindow, so the total must never become its expected area.
    plan={'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
          'sites':{'AU_AMRUN202309':site},'scene_interval':'2023-08-20/2023-10-15',
          'scene_cutoffs':{'pre_latest':'2023-08-31','post_earliest':'2023-10-01'},
          'acquisition_rule':'latest full-cover scene throughAug31;earliest full-cover scene fromOct1;scenecloud<20%;noimage/modelscoreselection',
          'source_period':{'year':2023,'month':9,'monthly_record_date':'2023-09-30',
                           'actual_cleared_hectares':119.57,'area_scope':'entire Amrun project, not downloaded subwindow'},
          'evidence':[{'kind':'third-party independent compliance audit, commissioned by operator and auditor approved by regulator',
                       'file':str(EVIDENCE/'independent_audit_2025.pdf'),
                       'sha256':file_sha256(EVIDENCE/'independent_audit_2025.pdf'),
                       'PDF_pages_1based':[6,59,60], 'printed_pages':[3,43,44],
                       'audit_period':'since prior audit in April2022 throughMarch2025',
                       'audit_conducted':'2025-03-10/2025-03-24',
                       'supports':'actual September2023 land clearing with dated end-of-month records and monthly survey shapefile data inspected',
                       'does_not_support':'our project field trial, accident absence, slope safety, exact downloaded-window clearing boundary'},
                      {'kind':'operator annual report with actual survey tracking',
                       'file':str(EVIDENCE/'annual_compliance_2024.pdf'),
                       'sha256':file_sha256(EVIDENCE/'annual_compliance_2024.pdf'),
                       'PDF_pages_1based':[4,9],'period':'2023-05-12/2024-05-11',
                       'actual_cleared_hectares_May2023_April2024':1481}],
          'reserved_envelopes_excluded':old['reserved_envelopes_excluded'],
          'TR_COP2024_excluded_by_nonoverlapping_country_and_location':True,
          'same_Amrun_mine_group_as_existing_development':True,
          'training_sites':['AU_AMRUN202309'],'normal_holdout_sites':[],
          'new_failure_positives':0,'pixel_normal_labels_created':0,'all_annotations_pending':True,
          'external_feedback_about_our_project':0,'classifier_predictions_at_registration':0}
    out=OUT
    if full_window:
        out=OUT/'full_published_geometry_window_v2'
        site['site_id']='AU_AMRUN202309_FULL'
        bounds=site['source_geometry_bounds'];margin=.008
        site['bbox_wgs84']=[bounds[0]-margin,bounds[1]-margin,bounds[2]+margin,bounds[3]+margin]
        plan['sites']={'AU_AMRUN202309_FULL':site};plan['training_sites']=['AU_AMRUN202309_FULL']
        plan['max_native_window_dimension']=1600
        plan['spatial_extension_reason']='central pre-existing subwindow misses most published mining footprint;cover original published geometry plus fixed0.008degree margin'
        plan['parent_partial_window_plan_sha256']=file_sha256(OUT/'site_plan.json')
        plan['spatial_extension_selected_from_published_geometry_not_classifier_scores']=True
        plan['still_not_complete_Amrun_project_area_or_monthly_survey_polygon']=True
    out.mkdir(exist_ok=True);p=out/'site_plan.json'
    if p.exists():
        saved=json.loads(p.read_text(encoding='utf-8'))
        if any(saved[k]!=v for k,v in plan.items() if k!='registered_at'):raise ValueError('Frozen actual-work plan changed')
        return
    p.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Registered actual September2023 clearing context before imagery;no safety or failure pixel labels')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--full-window',action='store_true');args=parser.parse_args()
    run(full_window=args.full_window)
