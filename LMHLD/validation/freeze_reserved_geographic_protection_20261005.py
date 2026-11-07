"""Protect original reserved windows independently of classifier success or file encoding."""
from pathlib import Path
from datetime import datetime
import sys,json
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rasterio
from rasterio.warp import transform_bounds
from mine_data_protocol import registry,verify_assets,DATA
from model_registry import file_sha256
def run():
    verify_assets();value=registry();rows=[]
    for case_id in ('CL0002','KG0001','US_BC2021','TR_COP2024'):
        case=value['cases'][case_id]
        if case['model_predictions_run']!=0:raise ValueError('Reserved case already opened')
        path=(DATA/'quality_audit'/case_id/'exact_date_quality_v1_20261005/post_producer_BGRN_original_grid.tif') if case_id in ('CL0002','KG0001') else DATA/'reserved_candidates'/case_id/'post_BGRN_10m.tif'
        with rasterio.open(path) as f:bounds=list(transform_bounds(f.crs,'EPSG:4326',*f.bounds,densify_pts=21))
        digest=file_sha256(path)
        if case.get('protected_geographic_extent_wgs84') not in (None,bounds):raise ValueError('Frozen protected geographic region changed')
        case.update(protected_geographic_extent_wgs84=bounds,protected_geographic_extent_source=str(path),protected_geographic_source_sha256=digest,
                    geographic_protection_scope='entire reserved raw source window, conservative isolation only;not survey accident geometry')
        key=path.relative_to(DATA).as_posix();assets=value.setdefault('additional_frozen_assets',{})
        if key in assets and assets[key]!=digest:raise ValueError('Frozen geographic source changed')
        assets[key]=digest;rows.append({'case_id':case_id,'source':str(path),'sha256':digest,'bounds_wgs84':bounds})
    (DATA/'mine_case_registry.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');verify_assets(value)
    report={'created_at':datetime.now().isoformat(timespec='seconds'),'cases':rows,'reserved_model_predictions':0,
            'plain_PNG_without_location_cannot_be_geographically_identified':True,
            'changed_bytes_cropped_or_removed_event_tags_GeoTIFF_still_protected':True}
    (ROOT/'validation/reserved_geographic_protection_20261005.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Four reserved raw windows frozen for spatial upload isolation;detector0')
if __name__=='__main__':run()
