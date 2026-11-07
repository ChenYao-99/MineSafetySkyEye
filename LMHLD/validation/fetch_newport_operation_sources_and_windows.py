"""Independent observed planned works, separate from permit-only or safety-unknown examples."""
from pathlib import Path
from datetime import datetime
import json,sys,urllib.request,hashlib
from rasterio.warp import transform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from audit_auxiliary_geometry_split import reserved_envelopes
import fetch_arid_incident_windows as acquire
OUT=ROOT/'datasets/arid_accident_operation_20261004';SOURCE=OUT/'source_review'
URLS={'newport_independent_inspection_20231127.pdf':'https://www.essex.gov.uk/sites/default/files/2023-12/Newport%20Quarry%20-%2027.11.23%20-%20site%20monitoring.pdf',
      'newport_operator_planning_statement_202211.pdf':'https://docs.planning.org.uk/20221215/138/RMXLRRQN01O00/oqh55xf4ljrhiz46.pdf',
      'bgs_newport_geographic_identity.html':'https://webapps.bgs.ac.uk/memoirs/docs/B01637.html'}


def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def run():
    sources={}
    for name,url in URLS.items():
        path=SOURCE/name
        if not path.exists():
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30) as r:data=r.read(16*1024*1024+1)
            if len(data)>16*1024*1024 or (name.endswith('.pdf') and not data.startswith(b'%PDF')):raise ValueError('Wrong/big primary source')
            path.write_bytes(data)
        sources[name]={'url':url,'sha256':file_sha256(path),'bytes':path.stat().st_size}
    # BGS identifies the chalk quarry at TL525331, not other Newport sites.
    lon,lat=transform('EPSG:27700','EPSG:4326',[552500],[233100]);lon,lat=lon[0],lat[0]
    bbox=[lon-.020,lat-.015,lon+.020,lat+.015]
    for b in reserved_envelopes().values():
        if bbox[0]<=b[2] and bbox[2]>=b[0] and bbox[1]<=b[3] and bbox[3]>=b[1]:raise ValueError('Protected mine overlap')
    c={'mine_group':'UK_NEWPORT_CHALK','role':'auxiliary_development','scope':'regulator_observed_planned_restoration_and_material_deposition',
       'inspection_date':'2023-11-27','event_source':URLS['newport_independent_inspection_20231127.pdf'],
       'location_source':URLS['bgs_newport_geographic_identity.html'],'source_national_grid_reference':'TL525331',
       'reference_scope':'independent officer recorded phase1 restoration and phase2a deposition in line with approved phasing; not a geotechnical safety certificate',
       'bbox_wgs84':bbox,'crs':'EPSG:32631','time_interval':'2023-08-15T00:00:00Z/2023-12-20T23:59:59Z',
       'pre_through':'2023-09-10','post_from':'2023-11-27','post_target':'2023-11-28',
       'normal_source_backing':'independent physical site visit, actual ongoing works recorded, not a future permit proposal',
       'explicit_no_failure_or_safety_statement':False,'pixel_negative_labels_automatically_admissible':False,
       'other_compliance_issues':'office and weighbridge locations differed slightly; not proof all activities conform'}
    p={'registered_at':datetime.now().isoformat(timespec='seconds'),'cases':{'UK_NEWPORT2023':c},'sources':sources,
       'scene_choice_rule':'fullAOI, scenecloud<20%; latest eligible throughSep10; post closest toNov28 after inspectionNov27; no model scores',
       'project_expert_feedback':0,'model_predictions':0,'no_accident_event_asserted':True}
    path=OUT/'newport_operation_plan.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'));p['registered_at']=old['registered_at']
        if old!=p:raise ValueError('Frozen operation plan changed')
    else:write(path,p)
    acquire.PLAN=path;acquire.run()


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
