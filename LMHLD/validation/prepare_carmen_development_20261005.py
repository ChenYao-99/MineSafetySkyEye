"""Register an additional pit-wall development mine before measurements/predictions."""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from model_registry import file_sha256
from validation.mine_data_protocol import registry,verify_assets
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
OUT=ROOT/'datasets/carmen_development_20261005'
SOURCE=ROOT/'datasets/pixel_boundary_sources_20261004/wang_supplement_original_media/image17.png'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
    verify_assets();OUT.mkdir(exist_ok=True)
    plan={'created_at':datetime.now().isoformat(timespec='seconds'),'datasets_drive':'D','reserved_inputs_or_predictions':0,
          'scene_choice_rule':'Only published SentinelVision exact pre15Dec2020 and post29Jan2021 dates;latest processing baseline then ID;fail rather than select by model results',
          'source_figure':str(SOURCE),'source_figure_sha256':file_sha256(SOURCE),
          'sources':{'government_event':'https://www.pna.gov.ph/articles/1125467',
                     'primary_InSAR_study':'https://www.e3s-conferences.org/articles/e3sconf/pdf/2023/52/e3sconf_dfhm82023_05020.pdf',
                     'mapped_outline':'https://ars.els-cdn.com/content/image/1-s2.0-S1569843226001330-mmc1.docx',
                     'published_optical_dates':'https://www.sentinelvision.eu/gallery/pdf/903ee6d97c404e74823719c9899ac053',
                     'pre_event_instability':'https://blogs.agu.org/landslideblog/2020/12/26/the-carmen-copper-mine-landslide-in-the-philippines-the-slope-evolution-before-failure/'},
          'normal_or_stable_negative_claim_allowed':False,'original_professional_survey_boundary_obtained':False,
          'figure_base_capture_date':None,'source_figure_pixels_are_not_model_input':True,
          'target_scope':'published accident core only, pending spatial holdout registration and observation review;no full accident IoU',
          'cases':{'PH_CARM2020':{'mine_group':'PH_CARMEN_TOLEDO','role':'development','event_date':'2020-12-21',
                                 'scope':'open_pit_wall_failure','bbox_wgs84':[123.690,10.315,123.756,10.354],
                                 'crs':'EPSG:32651','time_interval':'2020-12-15T00:00:00Z/2021-01-29T23:59:59Z',
                                 'pre_through':'2020-12-20','post_from':'2020-12-22','post_target':'2021-01-29'}}}
    path=OUT/'data_plan.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if any(old[k]!=v for k,v in plan.items() if k!='created_at'):raise ValueError('Frozen new mine plan changed')
    else:write(path,plan)
    value=registry();case=value['cases'].get('PH_CARM2020')
    if case is None:
        value['cases']['PH_CARM2020']={**plan['cases']['PH_CARM2020'],'mine_id':'PH_CARMEN_TOLEDO','quality_review_status':'pending',
            'acquisition':str(OUT/'PH_CARM2020/acquisition.json'),'model_predictions_run':0,'historical_event_model_exposure':False,
            'project_field_trial':False,'independent_expert_or_field_audit':False,'no_new_blind_test_claim':True,'source_plan':str(path)}
        write(ROOT/'datasets/mine_benchmark_20261002/mine_case_registry.json',value)
    elif case['role']!='development' or case['mine_group']!='PH_CARMEN_TOLEDO':raise ValueError('New mine role conflict')
    session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=(429,500,502,503,504))))
    ledger=OUT/'primary_source_access.json';rows=json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else {}
    url=plan['sources']['primary_InSAR_study'];target=OUT/'ramirez2023_carmen_primary.pdf'
    if not target.exists():
        try:
            r=session.get(url,timeout=(10,30));r.raise_for_status()
            if not r.content.startswith(b'%PDF') or len(r.content)>20*1024*1024:raise ValueError('Expected bounded primary PDF')
            target.write_bytes(r.content);rows['primary_study']={'url':url,'path':str(target),'sha256':file_sha256(target),'http_status':r.status_code}
        except Exception as e:rows['primary_study']={'url':url,'obtained':False,'error':str(e)}
        write(ledger,rows)
    original=ROOT/'validation/fetch_jagersfontein_original_pair.py';source=original.read_text(encoding='utf-8')
    replacements={"datasets/jagersfontein_development_20261005":"datasets/carmen_development_20261005", "['ZA_JAGS2022']":"['PH_CARM2020']",
                  "('2022-09-06','2022-09-16')":"('2020-12-15','2021-01-29')",
                  'Exact-date independentofficial-map developmentcase':'Exact-date source-backed pit-wall developmentcase'}
    for a,b in replacements.items():
        if a not in source:raise ValueError('Parent acquisition template changed')
        source=source.replace(a,b)
    dest=ROOT/'validation/fetch_carmen_original_pair_20261005.py'
    if dest.exists() and dest.read_text(encoding='utf-8')!=source:raise ValueError('Derived acquisition script changed')
    if not dest.exists():dest.write_text(source,encoding='utf-8')
    write(OUT/'acquisition_implementation_provenance.json',{'parent':str(original),'parent_sha256':file_sha256(original),
        'derived':str(dest),'derived_sha256':file_sha256(dest),'literal_replacements':replacements,'changes_to_raw_measurement_algorithm':False})
    print('Carmen2020 registered as development before acquisition;partial reference pending;no normal/safety claim',flush=True)
if __name__=='__main__':run()
