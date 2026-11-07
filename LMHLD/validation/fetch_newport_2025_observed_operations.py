"""New dated regulator visit, preserving the2023 cloud-rejected record separately."""
from pathlib import Path
from datetime import datetime
import json,sys,urllib.request
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
import fetch_arid_incident_windows as fetcher
OUT=ROOT/'datasets/arid_accident_operation_20261004';SRC=OUT/'source_review'
URL='https://www.essex.gov.uk/sites/default/files/2025-03/Newport%20Quarry-%20site%20monitoring%20-%2007.03.25.pdf'


def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def run():
    source=SRC/'newport_independent_inspection_20250307.pdf'
    if not source.exists():
        with urllib.request.urlopen(urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0'}),timeout=25) as r:data=r.read(2*1024*1024+1)
        if not data.startswith(b'%PDF') or len(data)>2*1024*1024:raise ValueError('Invalid regulator source')
        source.write_bytes(data)
    old=json.loads((OUT/'newport_operation_plan.json').read_text(encoding='utf-8'));base=old['cases']['UK_NEWPORT2023']
    case={**base,'inspection_date':'2025-03-07','event_source':URL,
       'reference_scope':'independent officer observed continuing phase1 restoration and deposition in phases2a/3; progress behind schedule, no recycling plant; no slope-safety certification',
       'time_interval':'2025-01-01T00:00:00Z/2025-04-20T23:59:59Z','pre_through':'2025-02-20','post_from':'2025-03-08','post_target':'2025-03-10',
       'normal_source_backing':'second dated independent onsite observation; no invented no-hazard conclusion',
       'same_mine_earlier2023_weather_rejection_retained':True}
    p={'registered_at':datetime.now().isoformat(timespec='seconds'),'cases':{'UK_NEWPORT2025':case},
       'scene_choice_rule':'fullAOI scenecloud<20%; latest throughFeb20 and earliest nearMar10 afterMar8; no model scores',
       'independent_inspection_sha256':file_sha256(source),'previous2023_source_plan_sha256':file_sha256(OUT/'newport_operation_plan.json'),
       'geotechnical_safety_truth':False,'pixel_negatives_automatically_admissible':False,'model_predictions':0}
    path=OUT/'newport2025_operation_plan.json'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf-8'));p['registered_at']=previous['registered_at']
        if previous!=p:raise ValueError('Frozen2025 operation plan changed')
    else:write(path,p)
    fetcher.PLAN=path;fetcher.run()


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
