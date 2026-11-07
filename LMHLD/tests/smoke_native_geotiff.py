# -*- coding: utf-8 -*-
"""Real current-weight API verification on a development-only native raster."""
import base64
import json
import sys
from pathlib import Path
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import detector_service as service


def run():
    source=Path('D:/LMHLD/datasets/mine_benchmark_20261002/development/LasCruces/post_BGRN_10m.tif')
    image=base64.b64encode(source.read_bytes()).decode(); client=service.app.test_client(); start=time.perf_counter()
    inspection=client.post('/api/inspect_upload',json={'image':image}); assert inspection.status_code==200,inspection.get_json()
    response=client.post('/api/detect_upload',json={'image':image,'model_tag':'mine','roi':[.25,.25,.9,.9]})
    assert response.status_code==200,response.get_json(); body=response.get_json()
    assert body['band_meta']['nir_source']=='measured'; assert body['scale_meta']['meters_per_pixel']==10
    assert body['quant']['ndvi']['available']; assert body['stats']['scope']=='roi'
    assert body['stats']['area_ratio']==body['roi']['risk_inside']
    export=None
    if body['stats']['regions']:
        r=client.post('/api/geometry_export',json={'regions':body['stats']['regions'],'format':'geojson','georeference':body['band_meta']['georeference']})
        assert r.status_code==200,r.get_json(); export=r.get_json()
        geo=json.loads(base64.b64decode(export['content_b64']))
        assert all(f['properties']['coordinate_source']=='uploaded_raster_header' for f in geo['features'])
    record={'case':'LasCruces development only','elapsed_seconds':round(time.perf_counter()-start,3),
            'NIR_source':body['band_meta']['nir_source'],'GSD':body['scale_meta'],'roi':body['roi'],
            'model':body['model_info'],'NDVI':body['quant']['ndvi'],'stats':body['stats'],
            'geojson_export':bool(export),'not_a_field_accuracy_benchmark':True,'reserved_test_predictions':0}
    out=Path('D:/LMHLD/experiments/geotiff_integration_20261003');out.mkdir(exist_ok=True)
    (out/'real_native_smoke.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('rgb_b64','overlay_b64'):(out/(name+'.png')).write_bytes(base64.b64decode(body[name]))
    if export:(out/'detection.geojson').write_bytes(base64.b64decode(export['content_b64']))
    print('Real GeoTIFF API smoke passed, seconds',record['elapsed_seconds'])


if __name__=='__main__':run()
