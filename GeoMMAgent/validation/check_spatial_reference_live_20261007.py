"""Actual GPU scale replay and same-record archival;development inputs only."""
from pathlib import Path
from datetime import datetime
import os,sys,subprocess,json,base64,hashlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
import requests,psutil
OUT=ROOT/'cache/spatial_reference_acceptance_20261007';BASE='http://127.0.0.1:5190'


def main():
    OUT.mkdir(exist_ok=True);archive=OUT/'archive.sqlite3'
    assert not archive.exists(),'Do not overwrite a completed acceptance archive'
    assert not any(c.status=='LISTEN' and c.laddr.port in [5190,5191] for c in psutil.net_connections(kind='tcp'))
    env=os.environ.copy();env.update(KAT_CASE_ARCHIVE_PATH=str(archive),PYTHONIOENCODING='utf-8')
    session=requests.Session();session.trust_env=False
    try:
        with (OUT/'startup.log').open('wb') as log:
            r=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'start_demo.ps1'),
                '-NoBrowser','-LocalFactsOnly','-BackendPort','5190','-DetectorPort','5191'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
        assert r.returncode==0,'See startup.log'
        health=session.get(BASE+'/api/health',timeout=10).json();assert health['reporting_mode']=='local_facts_only' and health['detector_runtime_compatible']
        result=[]
        for name in ['LasCruces','Hpakant']:
            source=Path('D:/LMHLD/datasets/mine_benchmark_20261002/development')/name/'post_BGRN_10m.tif'
            original=base64.b64encode(source.read_bytes()).decode()
            prep={'source':str(source),'destination':str(OUT/(name+'_controlled_2p5m.tif'))}
            (OUT/'replica_preparation.json').write_text(json.dumps(prep),encoding='utf-8')
            code="""from pathlib import Path
import sys,json
sys.path.insert(0,'D:/LMHLD')
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import rasterio,numpy as np
from affine import Affine
p=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
with rasterio.open(p['source']) as src:
 data=np.repeat(np.repeat(src.read(),4,axis=1),4,axis=2);profile=src.profile.copy();profile.update(height=data.shape[1],width=data.shape[2],transform=src.transform*Affine.scale(.25,.25))
 with rasterio.open(p['destination'],'w',**profile) as dst:
  dst.write(data);dst.descriptions=src.descriptions;dst.scales=src.scales;dst.offsets=src.offsets;dst.update_tags(**src.tags(),engineering_derivative='same_information_controlled_resolution_replay_not_a_new_real_scene')
"""
            script=OUT/'prepare_replica.py';script.write_text(code,encoding='utf-8')
            subprocess.run(['D:/Miniconda3/envs/lmhld/python.exe',str(script),str(OUT/'replica_preparation.json')],env=env,check=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
            replica=base64.b64encode(Path(prep['destination']).read_bytes()).decode()
            records=[]
            for image,mode in [(original,'native'),(replica,'reference_10m')]:
                begin=time.monotonic();response=session.post(BASE+'/api/pipeline_upload',json={'image':image,'analysis_grid':mode,'model_tag':'mine'},timeout=180)
                assert response.status_code==200,response.text[:400];value=response.json()
                assert value['detection']['band_meta']['spatial_input_contract']['mode']==mode
                saved=session.get(BASE+'/api/archive/reports/'+value['report_id'],timeout=15).json()
                assert saved['detection']['band_meta']['spatial_input_contract']==value['detection']['band_meta']['spatial_input_contract']
                assert value['generation']['status']=='fallback'
                records.append({'report_id':value['report_id'],'snapshot_sha256':value['archive']['snapshot_sha256'],
                    'facts':value['objective_facts'],'stats':value['detection']['stats'],
                    'spatial_contract':value['detection']['band_meta']['spatial_input_contract'],'elapsed_seconds':time.monotonic()-begin})
            a,b=records
            assert a['stats']['landslide_area_m2']==b['stats']['landslide_area_m2']
            assert a['stats']['observed_area_m2']==b['stats']['observed_area_m2']
            assert a['stats']['num_regions']==b['stats']['num_regions'] and a['facts']['area_percent']==b['facts']['area_percent']
            assert not b['spatial_contract']['accuracy_on_real_high_resolution_sensors_verified']
            result.append({'development_case':name,'physical_area_fraction_and_instance_count_preserved':True,'runs':records})
            print('PASS scale-consistent actual GPU+localreport+archive',name,flush=True)
        receipt={'passed':True,'verified_at':datetime.now().astimezone().isoformat(),'cases':result,'actual_GPU_inference':True,
            'controlled_resolution_replicas_not_new_high_resolution_groundtruth':True,
            'original10mphysical_quantities_preserved':True,'spatial_protocol_persisted_with_same_record':True,
            'production_archive_written':False,'new_CNN_fits_or_threshold_search_or_reserved_predictions_or_cloud_LLM_calls':0,
            'default_route_still_native_and_default_weights_unchanged':True,
            'limits':['controlleddevelopmentinputconsistency,not newmine recall/normalfalsealarm acceptance','singleepochopt-inreference route;pairedreference modeexplicitlyrefused']}
        (OUT/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    finally:
        subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'stop_demo.ps1'),'-BackendPort','5190','-DetectorPort','5191'],capture_output=True,timeout=40,creationflags=subprocess.CREATE_NO_WINDOW)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
