"""Small public inventory metadata and spatial queries, never interpreted as verified mine labels."""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime
import requests
ROOT=Path(__file__).resolve().parents[1]/'datasets/mine_benchmark_20261002'
OUT=ROOT/'boundary_catalogue_leads_20261004'
BASE='https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/COOLR_Events_Polygons/FeatureServer'


def run():
    OUT.mkdir(exist_ok=True);session=requests.Session();log=[]
    def get(name,url,params=None):
        target=OUT/(name+'.json');record={'url':url,'params':params,'requested_at':datetime.now().isoformat(timespec='seconds')}
        try:
            if not target.exists():
                response=session.get(url,params=params,timeout=(10,25));response.raise_for_status()
                if len(response.content)>8*1024*1024:raise ValueError('Metadata cap exceeded')
                value=response.json()
                if 'error' in value:raise ValueError(str(value['error']))
                target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            value=json.loads(target.read_text(encoding='utf-8'))
            record.update(status='available',file=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            log.append(record);return value
        except Exception as exc:
            record.update(status='unavailable',error=str(exc));log.append(record);return None
    metadata=get('nasa_coolr_service',BASE,{'f':'json'})
    if metadata:
        for layer in metadata.get('layers',[])[:8]:
            lid=layer['id'];get(f'nasa_coolr_layer{lid}',BASE+f'/{lid}',{'f':'json'})
            value=get(f'bingham_spatial_layer{lid}',BASE+f'/{lid}/query',{
                'f':'geojson','where':'1=1','geometry':'-112.20,40.48,-112.10,40.56','geometryType':'esriGeometryEnvelope',
                'inSR':4326,'spatialRel':'esriSpatialRelIntersects','outSR':4326,'outFields':'*','returnGeometry':'true','resultRecordCount':1000})
            if value:print(f'NASA layer{lid}: {len(value.get("features",[]))} spatial candidates; date/semantics not accepted',flush=True)
    for record in (20925213,14195737):
        value=get(f'zenodo_{record}',f'https://zenodo.org/api/records/{record}')
        if value:print(f'Zenodo{record}: '+json.dumps([{'key':f['key'],'bytes':f['size']} for f in value.get('files',[])]),flush=True)
    utah='https://services1.arcgis.com/99lidPhWCzftIe9K/ArcGIS/rest/services/LandslideInventoryPolygons/FeatureServer/0'
    get('utah_inventory_layer',utah,{'f':'json'})
    candidates=get('utah_bingham_spatial',utah+'/query',{
        'f':'geojson','where':'1=1','geometry':'-112.20,40.48,-112.10,40.56','geometryType':'esriGeometryEnvelope',
        'inSR':4326,'spatialRel':'esriSpatialRelIntersects','outSR':4326,'outFields':'*','returnGeometry':'true','resultRecordCount':1000})
    if candidates:print('Utah spatial candidates: '+str(len(candidates.get('features',[])))+'; official source last update2018 cannot certify2021 accident',flush=True)
    summary={'queries':log,'new_verified_accident_boundaries':0,'new_expert_reviews':0,'model_predictions':0,
             'all_geometries_pending_identity_date_method_license_review':True}
    (OUT/'acquisition.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'available':sum(r['status']=='available' for r in log),'queries':len(log),'new_verified_boundaries':0}))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
