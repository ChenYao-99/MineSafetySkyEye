# -*- coding: utf-8 -*-
"""Inspect originals and optical quality only; no masks or detector are created."""
import json
import sys
import cv2
import numpy as np
import rasterio
from mine_data_protocol import DATA, assert_development_cases
from register_las_cruces_boundary import rgb


def run():
    assert_development_cases(['Hpakant'])
    folder=DATA/'development/Hpakant'
    acquisition=json.loads((folder/'acquisition.json').read_text(encoding='utf-8'))
    images=[]; audit={'case_id':'Hpakant','role':'development_candidate_only','native_model_predictions_run':0,
        'native_boundary_georeferencing_complete':False,'independent_expert_review':False,
        'SCL_7_is_not_confirmed_visible':True,'SCL_3_shadow_retained':True,'images':{}}
    for period in ('pre','post'):
        images.append(rgb(folder/f'{period}_BGRN_10m.tif'))
        with rasterio.open(folder/f'{period}_SCL_10m.tif') as src: quality=src.read(1)
        # Approximate accident vicinity from the authors Figure1 graticule. This
        # rectangle is a visibility check, NEVER a positive accident label.
        with rasterio.open(folder/f'{period}_BGRN_10m.tif') as src:
            from rasterio.warp import transform
            east,north=transform('EPSG:4326',src.crs,[96.25,96.27],[25.47,25.482])
            inverse=~src.transform
            a,b=inverse*(east[0],north[0]),inverse*(east[1],north[1])
        x0,x1=sorted([max(0,int(a[0])),min(quality.shape[1],int(b[0]))])
        y0,y1=sorted([max(0,int(b[1])),min(quality.shape[0],int(a[1]))])
        vicinity=quality[y0:y1,x0:x1]
        audit['images'][period]={'datetime':acquisition['images'][period]['datetime'],
            'whole_ROI_optical_keep_fraction':float((~np.isin(quality,[0,1,8,9,10,11])).mean()),
            'approximate_vicinity_bbox_native_xyxy':[x0,y0,x1,y1],
            'approximate_vicinity_not_an_accident_truth_mask':True,
            'vicinity_SCL_counts':{str(int(v)):int(n) for v,n in zip(*np.unique(vicinity,return_counts=True))},
            'vicinity_optical_keep_fraction':float((~np.isin(vicinity,[0,1,8,9,10,11])).mean()),
            'source_date_note':'PreMay10 vs author PlanetMay11; postAug3 same calendar day, different sensor/time'}
    cv2.imencode('.png',np.hstack(images))[1].tofile(str(folder/'native_pre_post_RGB.png'))
    (folder/'candidate_quality.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8'); run()
