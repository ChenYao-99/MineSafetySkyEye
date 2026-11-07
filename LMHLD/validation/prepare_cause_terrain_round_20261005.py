"""Freeze a different supervised task and new physical context before any new fits."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from model_registry import file_sha256
from mine_data_protocol import registry,reserved_case_ids,assert_development_cases,verify_assets
OUT=ROOT/'experiments/cause_terrain_round_20261005';OUT.mkdir(exist_ok=True)
MAIN=['Amyntaio','LasCruces','Hpakant','CN_XJ2023']
FOLDERS={**{c:ROOT/'datasets/mine_benchmark_20261002/development'/('Xinjing' if c=='CN_XJ2023' else c) for c in MAIN},
 'ZA_JAGS2022':ROOT/'datasets/jagersfontein_development_20261005/ZA_JAGS2022',
 'ZA_GAMS2020S':ROOT/'datasets/arid_accident_operation_20261004/ZA_GAMS2020S',
 **{c:ROOT/'datasets/diverse_operation_pairs_20261004'/c for c in ('Amrun','Carajas','Faxe')},
 'UK_NEWPORT2025':ROOT/'datasets/arid_accident_operation_20261004/UK_NEWPORT2025',
 'Amrun_annual_SLATS2324':ROOT/'datasets/slats_operation_development_20261005/AU_AMRUN_SLATS2324'}

def run():
    assert_development_cases(MAIN+['ZA_JAGS2022','ZA_GAMS2020S']);verify_assets();v=registry()
    assert all(v['cases'][c]['model_predictions_run']==0 for c in reserved_case_ids(v))
    p={'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
       'purpose':'test independent accident/work-change supervision and historical terrain against a matched binary temporal CNN',
       'main_observed_development_mines':MAIN,'folds':[{'held':c,'train':[x for x in MAIN if x!=c]} for c in MAIN],
       'additional_development_sources':{c:str(path) for c,path in FOLDERS.items() if c not in MAIN},
       'all_sources':{c:str(path) for c,path in FOLDERS.items()},
       'all_main_mines_already_observed_not_new_independent_tests':True,
       'reserved_Morenci_Caval_failed_Kef_and_unadmitted_polygons_not_consumed':True,
       'arms':['binary_temporal','independent_cause_heads','independent_cause_heads_with_historical_terrain'],
       'architecture':'same small native96pixel temporal UNet,two sigmoid heads in all arms;no supervised mine checkpoint initialization',
       'new_information':'measured calibrated BGRN pair + per-epoch support;DEM context only in thirdarm',
       'supervision_contrast':'binary control maps government expectedwork to accidentnegative;independent arms give that source no accidentlabel and supervised workchange only;accident andwork may overlap',
       'work_change_negative_semantics':'only reviewed constructed surfaces with common valid observation and small measured spectral difference;not accidentpositive automatically treated as nonwork',
       'unknown_target255_has_zero_weight':True,'new_pixel_truth_or_professional_safety_claim':False,
       'terrain':{'collection':'cop-dem-glo-30','source':'https://planetarycomputer.microsoft.com/api/stac/v1/collections/cop-dem-glo-30',
                  'provider_description':'https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM',
                  'approximate_native_spacing_m':30,'source_acquisition_period':'2011-2015 with possible older infill',
                  'represents':'historical surface including vegetation/infrastructure;not contemporary pitwall or deformation measurement',
                  'native_original_window_retained':True,'projected_analysis_spacing_m':30,
                  'coarse_features_shared_at10m_optical_pixels_not_new10m_height_observations':True,
                  'absolute_elevation_coordinates_or_mine_ID_not_classifier_features':True},
       'seed':42,'planned_primary_new_CNN_fits':12,'native_crop_pixels':96,
       'training':{'epochs':10,'updates_per_epoch':48,'batch_size':8,'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,
                   'checkpoint':'fixed last epoch,noheldlabelsorresultsforselection','work_loss_weight':.3},
       'primary_accident_threshold':.5,'primary_work_threshold':.5,
       'uncertainty_reporting':'bothscores high -> competingcause review;abstentions not scored as background or hidden recall',
       'gates':{'macro_recall_min':.7,'worst_recall_min':.5,'reviewed_work_activation_max':.15,'weak_background_max':.15},
       'gates_are_internal_engineering_not_official_award_requirement':True,
       'larger_seed_or_architecture_sweep_only_if_primary_joint_gates_pass':True,
       'source_regression_and_runtime_required_before_any_default_promotion':True,
       'production_default_changed':False,
       'source_acceptance_sha256':file_sha256(ROOT/'datasets/slats_operation_development_20261005/work_reference_acceptance.json')}
    path=OUT/'round_plan.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if any(old.get(k)!=value for k,value in p.items() if k!='registered_at'):raise ValueError('Frozen cause/terrain experiment plan changed')
    else:path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('12 bounded matched CNN fits registered;dual independently supervised causes;historical terrain ablation;reserves untouched')

if __name__=='__main__':run()
