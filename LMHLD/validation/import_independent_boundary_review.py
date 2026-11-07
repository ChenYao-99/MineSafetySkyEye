# -*- coding: utf-8 -*-
"""Validate real review exports into quarantine; never auto-certify experts or train."""
from pathlib import Path
import json,hashlib,argparse,sys
import numpy as np
from mine_data_protocol import DATA,assert_development_cases

ALLOWED={'Amyntaio','LasCruces','Hpakant'}
KINDS={'uncertain','candidate_failure','normal_operation'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def simple_polygon(points):
    pts=np.asarray(points,np.float64)
    if pts.ndim!=2 or pts.shape[1]!=2 or not 3<=len(pts)<=500 or not np.isfinite(pts).all():raise ValueError('Invalid finite polygon vertices')
    if np.array_equal(pts[0],pts[-1]):pts=pts[:-1]
    if len(np.unique(pts,axis=0))!=len(pts):raise ValueError('Duplicate polygon sites')
    def cross(a,b,c):
        u,v=b-a,c-a
        return float(u[0]*v[1]-u[1]*v[0])
    def on(a,b,c):return abs(cross(a,b,c))<1e-8 and np.all(c>=np.minimum(a,b)-1e-8) and np.all(c<=np.maximum(a,b)+1e-8)
    n=len(pts)
    for i in range(n):
        a,b=pts[i],pts[(i+1)%n]
        for j in range(i+1,n):
            if j==i+1 or (i==0 and j==n-1):continue
            c,d=pts[j],pts[(j+1)%n];s1,s2,s3,s4=cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
            if (s1*s2<0 and s3*s4<0) or any((on(a,b,c),on(a,b,d),on(c,d,a),on(c,d,b))):raise ValueError('Self-intersecting polygon')
    area=abs(float(np.dot(pts[:,0],np.roll(pts[:,1],1))-np.dot(pts[:,1],np.roll(pts[:,0],1))))/2
    if area<1:raise ValueError('Polygon has insufficient area')
    return pts,area


def validate(record):
    if record.get('schema')!='mine_independent_review_v1':raise ValueError('Unsupported review schema')
    if record.get('automated_smoke_not_real_review') or 'AUTOMATED_SMOKE' in record.get('reviewer',''):raise ValueError('Automated test is not real review feedback')
    if not str(record.get('reviewer','')).strip():raise ValueError('Actual reviewer identifier required')
    if record.get('model_outputs_shown_by_tool') is not False or record.get('author_contours_shown_by_tool') is not False:raise ValueError('First-stage source-only exposure record required')
    ids=[c.get('case_id') for c in record.get('cases',[])]
    if not ids or len(set(ids))!=len(ids) or not set(ids).issubset(ALLOWED):raise ValueError('Only explicit development review cases allowed; reserved events forbidden')
    assert_development_cases(ids);features=[]
    for item in record['cases']:
        eid=item['case_id'];folder=DATA/f'development/{eid}'
        acquisition=json.loads((folder/'acquisition.json').read_text(encoding='utf-8'));post=acquisition['images']['post']
        if item.get('post_sha256')!=sha(folder/'post_BGRN_10m.tif'):raise ValueError('Review refers to different image bytes')
        if item.get('crs')!=post['crs'] or item.get('shape')!=post['shape'][1:] or not np.allclose(item.get('transform'),post['transform'],atol=1e-8,rtol=0):raise ValueError('Review image geometry differs')
        if item.get('post_date')!=post['datetime']:raise ValueError('Review acquisition date differs')
        t=np.array(item['transform']);h,w=item['shape']
        for polygon in item.get('polygons',[]):
            if polygon.get('kind') not in KINDS:raise ValueError('Unknown review class')
            pts,area=simple_polygon(polygon['vertices_pixel'])
            if np.any(pts<0) or np.any(pts[:,0]>w) or np.any(pts[:,1]>h):raise ValueError('Review polygon outside source raster')
            projected=np.stack([t[0]*pts[:,0]+t[1]*pts[:,1]+t[2],t[3]*pts[:,0]+t[4]*pts[:,1]+t[5]],1)
            supplied=np.asarray(polygon.get('vertices_projected'),np.float64)
            if len(supplied)==len(pts)+1 and np.array_equal(supplied[0],supplied[-1]):supplied=supplied[:-1]
            if supplied.shape!=projected.shape or not np.allclose(supplied,projected,atol=.001,rtol=0):raise ValueError('Pixel and projected vertices disagree')
            from rasterio.warp import transform
            lon,lat=transform(item['crs'],'EPSG:4326',projected[:,0].tolist(),projected[:,1].tolist());ring=np.stack([lon,lat],1).tolist();ring.append(ring[0])
            features.append({'type':'Feature','properties':{'case_id':eid,'review_class':polygon['kind'],'pixel_area':area,
                'expert_identity_verified':False,'independent_status':'pending_project_verification','no_automatic_training_use':True,
                'source_sha256':item['post_sha256']},'geometry':{'type':'Polygon','coordinates':[ring]}})
    if not features and not str(record.get('notes','')).strip():raise ValueError('No actual review geometry or notes')
    return {'type':'FeatureCollection','features':features,'properties':{'status':'quarantined_real_submission_pending_identity_independence_geometry_review',
        'expert_identity_verified':False,'input_identity_claims_not_trusted':True,
        'independence_self_report':record.get('previous_prediction_or_reference_exposure','unknown'),
        'new_verified_expert_reviews':0,'frozen_training_labels_modified':False}}


def run(path):
    path=Path(path);record=json.loads(path.read_text(encoding='utf-8-sig'));converted=validate(record)
    out=DATA/'independent_review_inbox'/sha(path)[:16];out.mkdir(parents=True,exist_ok=True)
    if not (out/'source_review.json').exists():(out/'source_review.json').write_bytes(path.read_bytes())
    (out/'review_geometries_wgs84.geojson').write_bytes((json.dumps(converted,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print('Review quarantined for actual identity/independence verification; verified experts0; no labels overwritten')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('review_json');run(p.parse_args().review_json)
