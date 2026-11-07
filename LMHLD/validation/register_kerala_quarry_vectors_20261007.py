"""Original NRSC mapped outlines joined to independently published quarry events."""
from pathlib import Path
from datetime import datetime
import hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'cache/research_dependencies')]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from shapely.geometry import shape
OUT=ROOT/'datasets/accident_boundary_pursuit_20261007/Kerala2018'
GROUPS={'IN_KERALA2018_QUU_ROW328':'IN_KERALA2018_KOTTAYAM',
    'IN_KERALA2018_QUU_ROW530':'IN_KERALA2018_KOZHIKODE','IN_KERALA2018_QUU_ROW531':'IN_KERALA2018_KOZHIKODE',
    'IN_KERALA2018_QUU_ROW1627':'IN_KERALA2018_CHITTUR','IN_KERALA2018_QUU_ROW1628':'IN_KERALA2018_ALATHUR'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    points=json.loads((OUT/'active_quarry_source_event_points.geojson').read_text(encoding='utf-8'))['features']
    rows=[];groups={}
    for point in points:
        key=point['id']
        if key not in GROUPS:continue
        path=OUT/(key+'_public_NRSC_feature_info.geojson');response=json.loads(path.read_text(encoding='utf-8'))
        assert response['crs']['properties']['name']=='urn:ogc:def:crs:EPSG::4326'
        assert len(response['features'])==1
        original=response['features'][0];geom=shape(original['geometry'])
        assert geom.geom_type in ['Polygon','MultiPolygon'] and not geom.is_empty and geom.is_valid
        source_area=original['properties']['area_m2'];author_area=float(point['properties']['Area'])
        difference=abs(source_area-author_area)
        assert difference<.01 and original['properties']['District']==point['properties']['District']
        assert point['properties']['NRSC']=='Y' and point['properties']['LU_2018']=='QUU'
        group=GROUPS[key];groups.setdefault(group,[]).append(point)
        rows.append({'source_event_id':key,'mine_group':group,'NRSC_original_feature_id':original['id'],
            'original_georeferenced_polygon_received':True,'original_provider_shape_unmodified':True,
            'official_area_m2':source_area,'independent_author_area_m2':author_area,'area_correspondence_difference_m2':difference,
            'district_correspondence':True,'author_GSI_field_survey_flag':point['properties']['GSI'],
            'published_landuse':'QUU,quarry in use','geometry_file':str(path),'geometry_file_sha256':sha(path),
            'individual_event_day_available':False,'nominal_event_period':'August2018KeralaMonsoon',
            'scope':'official mapped event outline,not a professional-ground-survey boundary or yet admitted exact pixel truth',
            'pixel_georegistration_and_observation_gates_passed':False,'new_classifier_predictions':0})
    plan_path=OUT/'native_pair_plan.json'
    if not plan_path.exists():
        cases={}
        for group,entries in groups.items():
            x=sum(z['geometry']['coordinates'][0] for z in entries)/len(entries)
            y=sum(z['geometry']['coordinates'][1] for z in entries)/len(entries)
            dx=750/(111320*math.cos(math.radians(y)));dy=750/111320
            cases[group]={'role':'reserved_final_test' if group.endswith('KOTTAYAM') else 'auxiliary_development',
                'classification_use_allowed_before_qualification':False,'bbox_wgs84':[x-dx,y-dy,x+dx,y+dy],
                'source_event_ids':[z['id'] for z in entries],
                'query_interval':'2018-01-01T00:00:00Z/2018-10-31T23:59:59Z',
                'pre_through':'2018-08-06','post_from':'2018-08-23','post_target':'2018-08-25',
                'post_event_period_reference':'source-wideAugust2018,not a fabricated individualday'}
        write(plan_path,{'registered_at':datetime.now().astimezone().isoformat(),'cases':cases,
            'source_metadata_and_polygon_inspection_before_roles':True,'roles_frozen_before_image_pixels_or_model_use':True,
            'selection_rule':'full-window cover,scenecloud<20%,latest eligible pre throughAug6;post closest toAug25 afterAug23;no model scores or geometry tuning',
            'native_multispectral_resampling_allowed':False,'max_pixels_per_side':180,'model_fits_or_predictions':0})
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    # Protect every new geographic window during qualification;only one is a future holdout.
    registry_path=ROOT/'datasets/input_reserve_registry.json';registry=json.loads(registry_path.read_text(encoding='utf-8'))
    for key,case in plan['cases'].items():
        matches=[r for r in rows if r['mine_group']==key]
        old=registry['cases'].get(key,{})
        extent=case['bbox_wgs84']
        if old.get('protected_geographic_extent_wgs84'):
            previous=old['protected_geographic_extent_wgs84']
            extent=[min(extent[0],previous[0]),min(extent[1],previous[1]),max(extent[2],previous[2]),max(extent[3],previous[3])]
        registry['cases'][key]={'role':'reserved_final_test','scope':'quarry-source geometry qualification only;classification blocked until admission;planned eventual role '+case['role'],
            'input_sha256_values':list(dict.fromkeys([r['geometry_file_sha256'] for r in matches]+old.get('input_sha256_values',[]))),
            'protected_geographic_extent_wgs84':extent,'planned_after_geometry_qualification_role':case['role']}
    write(registry_path,registry)
    write(OUT/'original_vector_correspondence.json',{'registered_at':datetime.now().astimezone().isoformat(),
        'original_polygon_records_received':len(rows),'distinct_geographic_groups':len(groups),'rows':rows,
        'original_NRSC_public_WMS_info_not_WFS_service_used':True,'publisher_vector_geometry_not_invented_from_points':True,
        'new_qualified_full_accident_pixel_boundaries':0,'no_model_predictions_or_professional_project_review':True})
    print('Original mapped polygons joined:',len(rows),'records,',len(groups),'groups;nativeQAroles frozen;all geographic inputs protected',flush=True)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
