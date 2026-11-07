"""Real loopback transport checks, with no model inference or cloud calls."""
from pathlib import Path
import base64
import hashlib
import json
import sys
import urllib.error
import urllib.request
ROOT=Path('D:/GeoMMAgent');sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
OUT=ROOT/'cache/engineering_integrity_20261006';OUT.mkdir(exist_ok=True)


def get(url):
    with urllib.request.urlopen(url,timeout=10) as response:
        return response.read()


def post(url,payload,declared_size=None):
    headers={'Content-Type':'application/json'}
    if declared_size:headers['Content-Length']=str(declared_size)
    request=urllib.request.Request(url,data=json.dumps(payload).encode('utf-8'),headers=headers)
    try:
        with urllib.request.urlopen(request,timeout=15) as response:
            return response.status,json.loads(response.read()),dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code,json.loads(error.read()),dict(error.headers)


def main():
    backend='http://127.0.0.1:5100';detector='http://127.0.0.1:5101'
    before=json.loads(get(detector+'/api/health'));assert not before['detector_loaded']
    health=json.loads(get(backend+'/api/health'));assert health['service']=='kat-backend' and health['detector_online']
    assert health['backend_code_sha256']==hashlib.sha256((ROOT/'backend.py').read_bytes()).hexdigest()
    assert get(backend+'/')==(ROOT/'矿安天眼_前端演示(后端版).html').read_bytes()
    checks=[]
    for host in (backend,detector):
        for payload,expected,declared in [([],400,None),({'image':'invalid bytes'},400,None),
                                          ({},413,48*1024*1024+1)]:
            status,body,headers=post(host+'/api/detect_upload',payload,declared)
            assert status==expected and body.get('error'),(host,status,body)
            if host==detector:assert headers.get('X-KAT-Request-ID')
            checks.append({'service':host,'status':status,'JSON_error':True})
    status,body,_=post(backend+'/api/pipeline',{'sample_index':True})
    assert status==400;checks.append({'service':backend,'status':status,'invalid_index_before_inference':True})
    source=Path('D:/LMHLD/datasets/mine_benchmark_20261002/development/LasCruces/post_BGRN_10m.tif')
    status,info,_=post(backend+'/api/inspect_upload',{'image':base64.b64encode(source.read_bytes()).decode('ascii')})
    assert status==200 and info['preview_b64'] and info['band_meta']['nir_source']=='measured'
    checks.append({'service':backend,'status':status,'actual_development_GeoTIFF_preview':True,
                   'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    after=json.loads(get(detector+'/api/health'));assert not after['detector_loaded']
    receipt={'passed':True,'real_HTTP_checks':checks,'served_frontend_matches_canonical':True,
             'current_backend_identity_verified':True,'before_detector_loaded':before['detector_loaded'],
             'after_detector_loaded':after['detector_loaded'],'new_classifier_predictions':0,'new_cloud_calls':0,
             'request_413_test_sends_only_declared_oversize_header_not_large_data':True}
    (OUT/'live_HTTP_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS 8 real HTTP checks, current served source, native GeoTIFF inspection, no model loading/inference')


if __name__=='__main__':main()
