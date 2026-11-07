"""New dated light/arid pit-wall incident development group, before measurements."""
from pathlib import Path
from datetime import datetime
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from model_registry import file_sha256
from mine_data_protocol import registry,verify_assets,DATA,assert_development_cases
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
OUT=ROOT/'datasets/kef_essnoun_development_20261005';SOURCE=ROOT/'datasets/pixel_boundary_sources_20261004/wang_supplement_original_media/image29.png'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
    verify_assets();OUT.mkdir(exist_ok=True)
    plan={'created_at':datetime.now().isoformat(timespec='seconds'),'datasets_drive':'D',
       'scene_choice_rule':'Nearest fully covering dates strictly before/after7Sep2022;latest same-date processing baseline/ID;no detector scores',
       'source_figure':str(SOURCE),'source_figure_sha256':file_sha256(SOURCE),'source_base_capture_date':None,
       'primary_field_study':'https://link.springer.com/article/10.1007/s10706-024-03027-5',
       'primary_source_scope':'publicabstract andpublisherexposedfigures only;no paid fulltext or originalsurveyGIS obtained',
       'event_identity':'7Sep2022 northwestern failure;do not conflate with2007 larger southernfailure',
       'professional_article_field_investigation_is_not_our_project_field_trial':True,
       'reference_readiness':'pending stable-ground spatial holdouts/chronology/actualvisibility;outside any acceptedcore unknown',
       'normal_or_geotechnically_safe_preimage_claim_allowed':False,'new_unseen_test_mine_claim':False,
       'cases':{'DZ_KEF2022':{'mine_group':'DZ_KEF_ESSNOUN','role':'development','scope':'open_pit_wall_failure','event_date':'2022-09-07',
              'material':'phosphate/marl;lightarid setting','bbox_wgs84':[7.932,34.676,7.999,34.741],
              'crs':'EPSG:32632','time_interval':'2022-08-25T00:00:00Z/2022-09-30T23:59:59Z',
              'pre_through':'2022-09-06','post_from':'2022-09-08','post_target':'2022-09-08'}}}
    path=OUT/'data_plan.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if any(old[k]!=v for k,v in plan.items() if k!='created_at'):raise ValueError('Frozen candidate plan changed')
    else:write(path,plan)
    value=registry()
    if 'DZ_KEF2022' not in value['cases']:
        value['cases']['DZ_KEF2022']={**plan['cases']['DZ_KEF2022'],'mine_id':'DZ_KEF_ESSNOUN',
              'quality_review_status':'pending','model_predictions_run':0,'project_field_trial':False,
              'independent_expert_or_field_audit':False,'new_blind_test_claim':False,'source_plan':str(path),
              'acquisition':str(OUT/'DZ_KEF2022/acquisition.json')}
        write(DATA/'mine_case_registry.json',value)
    assert_development_cases(['DZ_KEF2022'])
    parent=ROOT/'validation/fetch_jagersfontein_original_pair.py';text=parent.read_text(encoding='utf-8')
    text=text.replace('datasets/jagersfontein_development_20261005','datasets/kef_essnoun_development_20261005').replace("['ZA_JAGS2022']","['DZ_KEF2022']")
    first=text.index('            groups={d:');last=text.index('        before=sorted',first)
    text=text[:first]+"            eligible=[s for s in features if coverage(s)>=.999]\n"+text[last:]
    text=text.replace("key=lambda s:(s['properties']['datetime'],s['id']),reverse=True)","key=lambda s:(s['properties']['datetime'],str(s['properties'].get('s2:processing_baseline','')),s['id']),reverse=True)")
    text=text.replace("s['properties']['datetime'],s['id']))","s['properties']['datetime'], -float(s['properties'].get('s2:processing_baseline',0)),s['id']))")
    text=text.replace('Exact-date independentofficial-map developmentcase originalS2windows acquired','Chronology-fixed lightarid pitwall developmentcase originalS2windows acquired')
    target=ROOT/'validation/fetch_kef_original_pair_20261005.py'
    if target.exists() and target.read_text(encoding='utf-8')!=text:raise ValueError('Frozen derived native acquisition changed')
    if not target.exists():target.write_text(text,encoding='utf-8')
    write(OUT/'acquisition_implementation_provenance.json',{'parent':str(parent),'parent_sha256':file_sha256(parent),'derived':str(target),
             'derived_sha256':file_sha256(target),'specific_changes':'explicitnewdevelopmentcase;nearest covering chronology dates;baseline ties;no pixel/radiometryalgorithm change'})
    session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=2,backoff_factor=1,status_forcelist=(429,500,502,503,504))))
    records=[]
    for i,suffix in [(1,'png'),(2,'jpg'),(3,'png'),(4,'png')]:
        url=f'https://media.springernature.com/m312/springer-static/image/art%3A10.1007%2Fs10706-024-03027-5/MediaObjects/10706_2024_3027_Fig{i}_HTML.{suffix}'
        dst=OUT/f'publisher_public_preview_fig{i}.{suffix}'
        if dst.exists():records.append({'url':url,'path':str(dst),'sha256':file_sha256(dst),'purpose':'previewonly notmeasurementsorlabelgeometry'});continue
        try:
            r=session.get(url,timeout=(10,20));r.raise_for_status()
            if not r.headers.get('Content-Type','').startswith('image/') or len(r.content)>2*1024**2:raise ValueError('Expected bounded publicly exposed preview image')
            dst.write_bytes(r.content);records.append({'url':url,'path':str(dst),'sha256':file_sha256(dst),'purpose':'publisherpreviewonly;no inventednative/highresolutionpixels'})
        except Exception as e:records.append({'url':url,'obtained':False,'error':str(e)})
    write(OUT/'public_figure_access.json',records);print('Dated lightarid pitwallcase registered fordevelopment;publicfigures previewonly;pixelcore notyetaccepted')
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
