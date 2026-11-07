"""Freeze independently sourced auxiliary mines before images, annotations or predictions."""
from pathlib import Path
from datetime import datetime
import json,sqlite3,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'cache/research_dependencies'))
from shapely import from_wkb
from model_registry import file_sha256
from audit_auxiliary_geometry_split import reserved_envelopes
OUT=ROOT/'datasets/diverse_operation_pairs_20261004'
GEOM=ROOT/'datasets/mine_geometry_auxiliary_20261004'
SITES={
 'Morenci':{'tile_id':180,'role':'development_normal_holdout','source_split':'test','commodity':'copper',
            'material_context':'arid copper operation, mixed grey/brown constructed benches; optical colour is not mineral identification',
            'identity_source':'https://www.fcx.com/operations/north-america'},
 'Amrun':{'tile_id':2807,'role':'auxiliary_development_train','source_split':'train','commodity':'bauxite',
          'material_context':'bauxite operation with red/brown exposed soils and constructed haul infrastructure',
          'identity_source':'https://www.riotinto.com/en/operations/anz/weipa'},
 'Carajas':{'tile_id':944,'role':'auxiliary_development_train','source_split':'train','commodity':'iron ore',
            'material_context':'hematite operation in tropical terrain; actual surface colour will be visually reviewed',
            'identity_source':'https://www.vale.com/en/para'},
 'Faxe':{'tile_id':None,'role':'auxiliary_development_train','source_split':None,'commodity':'limestone',
         'material_context':'coral/bryozoan limestone and flint, light constructed quarry surfaces',
         'identity_source':'https://www.lhoist.com/da-DK/news-and-stories/story/with-x-ray-sorting-towards-a-more-sustainable-way-of-mining',
         'location_source':'https://www.faxekommune.dk/da/erhverv/miljoe/virksomheder/raastoffer',
         'approximate_center_wgs84':[12.115,55.255]}}


def run():
    OUT.mkdir(exist_ok=True);dest=OUT/'site_plan.json'
    if dest.exists():print('Reusing frozen4-site plan');return
    audit=json.loads((GEOM/'split_audit.json').read_text(encoding='utf-8'));db=sqlite3.connect((GEOM/'mining_area_data.gpkg').as_uri()+'?mode=ro',uri=True)
    sites={};env_sizes={0:0,1:32,2:48,3:48,4:64};protected=reserved_envelopes()
    for name,definition in SITES.items():
        row={**definition,'site_id':name,'source_geometry_not_normal_or_failure_truth':True,
             'image_or_model_scores_used_for_site_choice':False,'expert_feedback_received':0}
        if row['tile_id'] is not None:
            item=next(r for r in audit['all_rows'] if r['tile_id']==row['tile_id'])
            if item['split']!=row['source_split'] or not item['eligible_for_development_geometry']:raise ValueError('Source split or reserved envelope mismatch')
            layer=item['preferred_dataset']+'_polygons';blob=db.execute('select geom from '+layer+' where tile_id=?',(row['tile_id'],)).fetchone()[0]
            if blob is None or blob[:2]!=b'GP':raise ValueError('Missing real geometry')
            geom=from_wkb(blob[8+env_sizes[(blob[3]>>1)&7]:])
            pieces=list(geom.geoms) if geom.geom_type=='MultiPolygon' else [geom];largest=max(pieces,key=lambda g:g.area)
            center=[largest.centroid.x,largest.centroid.y]
            row.update(center_wgs84=center,coordinate_source='published actual largest contiguous mining-area polygon, verified source split',
                       original_geometry_table=layer,original_geometry_fid=item['fid'],source_geometry_bounds=list(largest.bounds))
        else:
            center=row['approximate_center_wgs84'];row.update(center_wgs84=center,coordinate_source='approximate map anchor; actual limestone quarry identity must pass original-image visual review')
        lon,lat=center;bbox=[lon-.025,lat-.02,lon+.025,lat+.02]
        if any(bbox[0]<=b[2] and bbox[2]>=b[0] and bbox[1]<=b[3] and bbox[3]>=b[1] for b in protected.values()):raise ValueError('New auxiliary mine overlaps protected surroundings')
        row.update(bbox_wgs84=bbox,crs=f'EPSG:{(32600 if lat>=0 else 32700)+int((lon+180)//6)+1}',
                   accident_event_asserted=False,normal_or_stability_labels_available=False,not_a_new_blind_accident=True)
        sites[name]=row
    p={'registered_at':datetime.now().isoformat(timespec='seconds'),'sites':sites,
       'original_GPKG_sha256':file_sha256(GEOM/'mining_area_data.gpkg'),'original_source_split_sha256':file_sha256(GEOM/'split_audit.json'),
       'reserved_envelopes_excluded':protected,'scene_interval':'2023-06-01/2023-07-15',
       'fixed_scene_rule':'latest full-cover scene throughJun15 and earliest no earlier thanJun25, scenecloud<20%; no model scores',
       'training_sites':['Amrun','Carajas','Faxe'],'normal_holdout_sites':['Morenci'],
       'holdout_policy':'entire Morenci group excluded from training and sampler; new appearance holdout, not independent accident final test',
       'all_annotations_pending':True,'new_failure_positives':0}
    dest.write_bytes((json.dumps(p,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({s:{'center':r['center_wgs84'],'role':r['role']} for s,r in sites.items()}))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
