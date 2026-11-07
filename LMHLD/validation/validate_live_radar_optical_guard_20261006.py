"""Isolated loopback HTTP smoke; never load a model or touch reserved rasters."""
from pathlib import Path
import sys,subprocess,socket,time,json,base64
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import requests
from model_registry import file_sha256
OUT=ROOT/'experiments/sar_spatial_qualification_20261006'

def run():
    code="""import sys
sys.path.insert(0,'D:/LMHLD')
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import detector_service as service
def forbidden_model(*args,**kwargs):
    raise AssertionError('Live sensor-semantics check must not load a detector')
service._load_upload_detector=forbidden_model
service.app.run(host='127.0.0.1',port=int(sys.argv[1]),debug=False,threaded=True,use_reloader=False)
"""
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    log=OUT/'live_sensor_semantics_smoke.log'
    rows=[]
    with log.open('wb') as handle:
        child=subprocess.Popen([sys.executable,'-c',code,str(port)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            base='http://127.0.0.1:'+str(port);deadline=time.monotonic()+45;health=None
            while time.monotonic()<deadline:
                if child.poll() is not None:raise RuntimeError('Isolated detector process exited before readiness')
                try:
                    r=requests.get(base+'/api/health',timeout=2)
                    if r.status_code==200:health=r.json();break
                except requests.RequestException:pass
                time.sleep(.5)
            if health is None:raise RuntimeError('Isolated detector service did not become ready')
            for name in ['pre/vv_original_raw_window.tif','radiometry_v1/pre_vv_sigma0.tif','terrain_correction_v1/pre_vv_gamma0_30m.tif']:
                path=ROOT/'datasets/sar_development_pairs_20261005/LasCruces'/name
                b64=base64.b64encode(path.read_bytes()).decode()
                for endpoint,body in [('/api/inspect_upload',{'image':b64}),('/api/detect_upload',{'image':b64}),('/api/compare',{'image_t1':b64,'image_t2':b64})]:
                    r=requests.post(base+endpoint,json=body,timeout=15);a=r.json()
                    assert r.status_code==400 and 'SAR' in a['error'] and 'risk_level' not in a,(name,endpoint,a)
                    rows.append({'source':name,'endpoint':endpoint,'status':r.status_code,'rejected_before_model':True})
            p=ROOT/'datasets/mine_benchmark_20261002/development/LasCruces/post_BGRN_10m.tif'
            r=requests.post(base+'/api/inspect_upload',json={'image':base64.b64encode(p.read_bytes()).decode()},timeout=15)
            assert r.status_code==200 and r.json()['band_meta']['nir_source']=='measured'
            rows.append({'source':'known_LasCruces_optical_development_original','endpoint':'/api/inspect_upload','status':200,'model_called':False})
            after=requests.get(base+'/api/health',timeout=5).json()
            assert not health.get('detector_loaded') and not after.get('detector_loaded')
        finally:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
    receipt={'tests':rows,'real_loopback_HTTP_checks_passed':10,'model_loader_forbidden_during_smoke':True,
        'default_detector_loaded':False,'temporary_hidden_process_terminated':True,'user_services_restarted_or_reconfigured':False,
        'reserved_inputs':0,'detector_predictions':0,'raster_upload_sha256':file_sha256(ROOT/'raster_upload.py'),
        'default_weight_sha256':file_sha256(ROOT/'models_prior/real_ft_T1b_ours_seed42/best_model.pth')}
    (OUT/'live_sensor_semantics_smoke.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('RealHTTP10checks passed;9SARrejections beforemodel;1opticalinspection;0models;temporaryprocessstopped',flush=True)

if __name__=='__main__':run()
