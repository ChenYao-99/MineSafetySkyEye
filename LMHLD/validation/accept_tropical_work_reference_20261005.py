"""Record original-image acceptance before fitting or viewing new predictions."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
OUT=ROOT/'datasets/tropical_work_pairs_20261005'
LOCAL=ROOT/'datasets/tropical_work_pairs_local_quality_20261005'
REFERENCES={
 'AU_ANDOOM_LOCAL_SLATS2324':[LOCAL/'AU_ANDOOM_LOCAL_SLATS2324/work_reference_v1_SLATS/reference.json'],
 'AU_BAUXITEHILLS_LOCAL_SLATS2324':[LOCAL/'AU_BAUXITEHILLS_LOCAL_SLATS2324/work_reference_v1_SLATS/reference.json'],
 'AU_MTCARBINE_SLATS2324':[OUT/'AU_MTCARBINE_SLATS2324/work_reference_v1_SLATS/reference.json',OUT/'AU_MTCARBINE_SLATS2324/constructed_road_reference_v1/reference.json'],
 'AU_CAPEFLATTERY_SLATS2324':[OUT/'AU_CAPEFLATTERY_SLATS2324/work_reference_v1_SLATS/reference.json'],
}

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def run():
    accepted={}
    for case,paths in REFERENCES.items():
        entries=[]
        for path in paths:
            row=load(path)
            for name,digest in row['frozen_assets'].items():assert file_sha256(path.parent/name)==digest
            assert row['classifier_predictions_used_for_annotation']==0
            if 'stable_ground_QA_not_ground_displacement' in row:
                assert row['stable_ground_QA_not_ground_displacement']['raw_pair_gate_pass'] and row['common_visible_fraction']>=.6
                count=row['expected_work_pixels'];kind='government annual mining clearing plus observed change;weak expectedwork'
            else:
                assert not row['certified_normal_year_or_accident_absence_truth'] and row['period_contains_cyclone_and_flooding']
                count=row['weak_constructed_appearance_pixels'];kind='project original-pair constructed road/bench appearance;weak normal reference'
            entries.append({'reference':str(path),'sha256':file_sha256(path),'pixels':count,'kind':kind,'confidence':.25})
        accepted[case]={'mine_group':row['mine_group'],'role':row['role'],'references':entries,
                        'visual_review':'original calibrated native pair inspected before any candidate predictions;selected source cores concur with rectangular mining clearance or persistent constructed access/bench appearance;weather and unobserved regions unknown',
                        'expert_endorsement_or_field_trial':False,'daily_operation_or_accident_absence_proven':False,
                        'unchanged40m_official_boundary_and1pixel_manual_road_edge_uncertainty':True}
    receipt={'reviewed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'accepted':accepted,
             'pending':{'AU_ANDOOM_SLATS2324':{'reason':'original scene pair commonvisible.531 below.6;excluded;localqualityrescue kept distinct', 'training_admitted':False},
                        'AU_BAUXITEHILLS_SLATS2324':{'reason':'no pre image under coarse<20%scenecloud;localqualityrescue kept distinct','training_admitted':False}},
             'new_accident_positive_pixels':0,'original_professional_accident_boundaries_added':0,
             'independent_annual_source_not_independent_pixel_safety_truth':True,'project_expert_feedback_received':0,
             'normal_holdout_full_region':'AU_CAPEFLATTERY_SLATS2324','holdout_pixels_not_used_to_fit_or_select_checkpoint':True,
             'classifier_predictions_used':0}
    path=OUT/'work_reference_acceptance.json'
    if path.exists():
        old=load(path);assert all(old.get(k)==v for k,v in receipt.items() if k!='reviewed_at')
    else:path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print({case:[entry['pixels'] for entry in row['references']] for case,row in accepted.items()})

if __name__=='__main__':run()
