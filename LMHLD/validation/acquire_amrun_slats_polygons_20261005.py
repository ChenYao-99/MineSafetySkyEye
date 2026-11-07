"""Original government change polygons only; no screenshots or safety inference."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,json,sys,collections
import requests
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from model_registry import file_sha256
OUT=ROOT/'datasets/actual_operation_evidence_20261005/Amrun/SLATS_official_spatial';OUT.mkdir(exist_ok=True)
PLAN=ROOT/'datasets/source_backed_operation_change_20261005/full_published_geometry_window_v2/site_plan.json'
SERVICE='https://spatial-gis.information.qld.gov.au/arcgis/rest/services/Environment/SLATS/MapServer'

def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def obtain(session,path,url,params):
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    r=session.get(url,params=params,timeout=(10,40));r.raise_for_status()
    if len(r.content)>30*1024*1024:raise ValueError('Expected bounded public spatial response')
    data=r.json()
    if data.get('error'):raise ValueError(str(data['error']))
    write(path,data);return data

def run(site_plan=PLAN,site_id='AU_AMRUN202309_FULL',output=OUT):
    parent=json.loads(site_plan.read_text(encoding='utf-8'));bbox=parent['sites'][site_id]['bbox_wgs84']
    output.mkdir(exist_ok=True)
    group=parent['sites'][site_id].get('mine_group','Amrun')
    proposal={'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'role':'expected-work development context',
              'site_id':site_id,'mine_group':group,
              'parent_spatial_plan_sha256':file_sha256(site_plan),'bbox_wgs84':bbox,'service':SERVICE,
              'nominal_annual_periods':['2022-2023','2023-2024'],'requested_layer_ids':[254,255],
              'geometry_rule':'entire independently published mining-footprint subwindow plus its predeclared margin;all feature classes,not selected from model responses',
              'raw_government_geometry_not_rendered_map_pixels':True,'new_or_blind_mine_claim':False,
              'government_landcover_replacement_is_not_slope_safety_or_month_specific_truth':True,
              'classifier_predictions_for_source_or_label_selection':0,'pixel_labels_accepted':False}
    path=output/'source_plan.json'
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        if any(saved.get(k)!=v for k,v in proposal.items() if k!='registered_at'):
            corrected=dict(saved);corrected.update(role=proposal['role'],site_id=site_id,mine_group=group)
            if any(corrected.get(k)!=v for k,v in proposal.items() if k!='registered_at'):raise ValueError('Frozen government source query changed')
            (output/'source_plan_before_identity_correction.json').write_bytes(path.read_bytes())
            write(output/'query_identity_correction.json',{'old_plan_sha256':file_sha256(path),
                  'reason':'add exact site/mine-group identity to query metadata;registered bounding box,periods andall original provider geometries unchanged',
                  'corrected_site_id':site_id,'corrected_mine_group':group})
            write(path,corrected)
    else:write(path,proposal)
    session=requests.Session();service=obtain(session,output/'service_metadata.json',SERVICE,{'f':'json'})
    records=[]
    geometry=json.dumps({'xmin':bbox[0],'ymin':bbox[1],'xmax':bbox[2],'ymax':bbox[3],'spatialReference':{'wkid':4326}})
    for layer_id in (254,255):
        base=SERVICE+'/'+str(layer_id);meta=obtain(session,output/(str(layer_id)+'_metadata.json'),base,{'f':'json'})
        if meta.get('type')!='Feature Layer' or meta.get('geometryType')!='esriGeometryPolygon':raise ValueError('Actual polygon feature layer required')
        params={'where':'1=1','geometry':geometry,'geometryType':'esriGeometryEnvelope','inSR':4326,
                'spatialRel':'esriSpatialRelIntersects','f':'json','returnIdsOnly':'true'}
        ids=obtain(session,output/(str(layer_id)+'_object_IDs.json'),base+'/query',params)
        identifiers=ids.get('objectIds') or []
        if len(identifiers)>2000:raise ValueError('Explicit pagination needed for unexpectedly large local result')
        params.update(f='geojson',returnIdsOnly='false',returnGeometry='true',outFields='*',outSR=4326,
                      resultRecordCount=2000)
        geo=obtain(session,output/(str(layer_id)+'_original_polygons.geojson'),base+'/query',params)
        if geo.get('exceededTransferLimit') or len(geo.get('features',[]))!=len(identifiers):raise ValueError('Incomplete original polygon query')
        field=ids.get('objectIdFieldName') or meta['objectIdField']
        if {f['properties'][field] for f in geo['features']}!=set(identifiers):raise ValueError('Original feature IDs differ from independent count query')
        categories=collections.Counter(str(f['properties'].get('descr',f['properties'].get('class'))) for f in geo['features'])
        records.append({'layer_id':layer_id,'name':meta['name'],'feature_count':len(identifiers),
                        'original_property_fields':[f['name'] for f in meta['fields']],
                        'categories':dict(categories),'original_polygons_sha256':file_sha256(output/(str(layer_id)+'_original_polygons.geojson')),
                        'metadata_sha256':file_sha256(output/(str(layer_id)+'_metadata.json'))})
        print(layer_id,meta['name'],'features',len(identifiers),'categories',dict(categories),flush=True)
    receipt={'acquired_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'source_plan_sha256':file_sha256(path),
             'provider':'Queensland Government SLATS','records':records,'all_geometries_complete_for_registered_query':True,
             'source_data_local_month_of_clearing_not_assumed':True,'new_field_trials_or_project_expert_feedback':0,
             'classifier_predictions':0,'accepted_normal_or_accident_pixel_labels':0}
    write(output/'acquisition.json',receipt)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site-plan',type=Path,default=PLAN)
    parser.add_argument('--site-id',default='AU_AMRUN202309_FULL');parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();run(args.site_plan,args.site_id,args.output)
