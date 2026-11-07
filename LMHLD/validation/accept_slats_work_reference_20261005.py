"""Record original-image review. Government work classification is not mine safety."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
DATA=ROOT/'datasets/slats_operation_development_20261005'

def run():
    accepted={};pending={}
    for case in ('AU_AMRUN_SLATS2324','AU_CAVAL_REGION_SLATS2324'):
        folder=DATA/case/'work_reference_v1_SLATS';path=folder/'reference.json'
        row=json.loads(path.read_text(encoding='utf-8'))
        for name,digest in row['frozen_assets'].items():assert file_sha256(folder/name)==digest
        assert row['classifier_predictions_used_for_annotation']==0
        passed=row['stable_ground_QA_not_ground_displacement']['raw_pair_gate_pass'] and row['common_visible_fraction']>=.6
        if case=='AU_AMRUN_SLATS2324':
            assert passed and row['expected_work_pixels']>=100
            accepted[case]={'reference':str(path),'reference_sha256':file_sha256(path),
                  'mine_group':row['mine_group'],'pixels':row['expected_work_pixels'],'confidence':.25,
                  'annotation_epoch':'post only;pre green overlay merely locates the future post-reference for visualcomparison',
                  'visual_review':'original calibrated native pre/post confirms wooded areas become rectilinear bare working areas connected to mine haul infrastructure inside government annual mine-clearing geometry;no inference output inspected',
                  'independent_annual_mine_clearing_class':True,'literal_daily_operation_date_verified':False,
                  'safety_or_absence_of_hidden_instability_verified':False,
                  'outside_source_geometry_boundary_band_and_unavailable_pixels_unknown':True,
                  'allowed_use':'weak expected-work context for development and reference activation;not strong pixel accident background truth or accident absence'}
        else:
            assert row['expected_work_pixels']==5
            pending[case]={'reference':str(path),'pixels':5,'reason':'annual mine polygons in thiswindow are mostly narrow;only5jointvisiblevegetationlossinteriorpixels;insufficient core for newtraining source',
                  'training_admitted':False,'threshold_or40m_uncertainty_band_relaxed':False}
    receipt={'reviewed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'accepted':accepted,'pending':pending,
             'new_independent_mine_count_not_equal_pixelcount':True,'expert_or_field_trial_about_project':False,'detector_predictions_used':0}
    (DATA/'work_reference_acceptance.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Accepted56449 lowweight source-supported annual expected-work pixels;Caval5pixelproposal remains untrained;no safetytruth claim')

if __name__=='__main__':run()
