"""Force-stop only our test worker and verify recovery using the normal launcher."""
from pathlib import Path
from datetime import datetime
import base64,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
import requests,psutil
OCCUPIED='--occupied-port' in sys.argv
AUTOMATIC='--automatic' in sys.argv or OCCUPIED
label='detector_occupied_port' if OCCUPIED else 'detector_automatic_recovery' if AUTOMATIC else 'detector_recovery'
OUT=Path(os.environ.get('KAT_RECOVERY_ACCEPTANCE_OUT',str(ROOT/'cache/engineering_hardening_20261007'/label))).resolve()
assert OUT.is_relative_to((ROOT/'cache').resolve())
BACKEND_PORT=5156 if OCCUPIED else 5136 if AUTOMATIC else 5126;DETECTOR_PORT=BACKEND_PORT+1
BASE=f'http://127.0.0.1:{BACKEND_PORT}';DET=f'http://127.0.0.1:{DETECTOR_PORT}'
session=requests.Session();session.trust_env=False


def main():
    OUT.mkdir(parents=True,exist_ok=True);archive=OUT/'archive.sqlite3'
    assert not archive.exists(),'Acceptance already ran;do not replace its records'
    assert not any(c.status=='LISTEN' and c.laddr.port in [BACKEND_PORT,DETECTOR_PORT] for c in psutil.net_connections(kind='tcp'))
    env=os.environ.copy();env.update(KAT_CASE_ARCHIVE_PATH=str(archive),PYTHONIOENCODING='utf-8',KAT_AUTO_RECOVER_DETECTOR='1' if AUTOMATIC else '0')
    command=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'start_demo.ps1'),
        '-NoBrowser','-LocalFactsOnly','-BackendPort',str(BACKEND_PORT),'-DetectorPort',str(DETECTOR_PORT)]
    def launch(environment,label):
        with (OUT/(label+'.log')).open('wb') as log:
            return subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW).returncode
    def health():return session.get(BASE+'/api/health',timeout=10).json()
    image=base64.b64encode(Path('D:/LMHLD/datasets/mine_benchmark_20261002/development/LasCruces/post_BGRN_10m.tif').read_bytes()).decode()
    def task():
        response=session.post(BASE+'/api/pipeline_upload',json={'image':image,'model_tag':'mine'},timeout=180)
        assert response.status_code==200,response.status_code
        return response.json()
    fake=None
    try:
        assert launch(env,'startup')==0
        original=health();first=task();det=session.get(DET+'/api/health',timeout=10).json()
        process=psutil.Process(det['runtime_pid']);commandline=' '.join(process.cmdline()).lower().replace('\\','/')
        assert 'd:/lmhld/detector_service.py' in commandline and det['runtime_root']=='D:\\LMHLD'
        process.kill();process.wait(timeout=15)
        if OCCUPIED:
            from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
            import threading
            calls=[]
            class OtherService(BaseHTTPRequestHandler):
                def do_GET(self):
                    content=json.dumps({'status':'ok','service':'unrelated-test-service'}).encode()
                    self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
                def do_POST(self):calls.append('POST');self.send_error(403)
                def log_message(self,*args):pass
            fake=ThreadingHTTPServer(('127.0.0.1',DETECTOR_PORT),OtherService)
            threading.Thread(target=fake.serve_forever,daemon=True).start();time.sleep(5)
            conflicting=health()
            assert not conflicting['detector_runtime_compatible'] and conflicting['detector_recovery']['recovery_attempts']==0
            assert conflicting['detector_recovery']['state']=='port_occupied'
            response=session.post(BASE+'/api/detect_upload',json={'image':image},timeout=15)
            assert response.status_code==503 and response.json()['inference_status']=='runtime_version_mismatch'
            assert not calls,'Image content was sent to an unrelated service'
            assert launch(env,'occupied_port_refused')!=0
            fake.shutdown();fake.server_close();fake=None
        if AUTOMATIC:
            deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                current=health()
                if current['detector_online'] and current['detector_runtime_compatible'] and current['detector_recovery']['recovery_attempts']>=1:break
                time.sleep(.5)
            else:raise AssertionError('Automatic recovery did not restore the owned detector')
            assert current['detector_recovery_enabled']
        else:
            assert not health()['detector_online']
            unavailable=session.post(BASE+'/api/detect_upload',json={'image':image},timeout=15)
            assert unavailable.status_code==503 and unavailable.json().get('error')
        assert launch(env,'recovery')==0
        recovered=health();new_det=session.get(DET+'/api/health',timeout=10).json()
        assert recovered['runtime_pid']==original['runtime_pid']
        assert new_det['runtime_pid']!=det['runtime_pid'] and recovered['detector_runtime_compatible']
        assert recovered['archive_database_path']==str(archive)
        second=task();assert second['objective_facts']==first['objective_facts']
        assert second['review_handoff']['trace']['weights_sha256']==first['review_handoff']['trace']['weights_sha256']
        saved=session.get(BASE+'/api/archive/reports/'+first['report_id'],timeout=15).json()
        assert saved['objective_facts']==first['objective_facts'] and saved['archive']['snapshot_sha256']==first['archive']['snapshot_sha256']
        changed=env.copy();changed['KAT_CASE_ARCHIVE_PATH']=str(OUT/'unactivated_archive.sqlite3')
        assert launch(changed,'different_archive_refused')!=0
        assert not Path(changed['KAT_CASE_ARCHIVE_PATH']).exists()
        assert health()['runtime_pid']==original['runtime_pid']
        receipt={'passed':True,'verified_at':datetime.now().astimezone().isoformat(),'backend_PID_preserved':original['runtime_pid'],
            'old_detector_PID':det['runtime_pid'],'new_detector_PID':new_det['runtime_pid'],
            'offline_request_explicit_503':not AUTOMATIC,'standard_launcher_recovers_stopped_detector':not AUTOMATIC,
            'automatic_owned_child_recovery':AUTOMATIC,'recovery_state':recovered.get('detector_recovery'),
            'occupied_foreign_service_left_running_and_no_payload_forwarded':OCCUPIED,
            'first_archive_facts_and_SHA_preserved':True,'new_GPU_prediction_same_default_and_objective_facts':True,
            'silent_switch_to_another_archive_refused':True,'production_archive_written':False,
            'runtime_identity':{'backend':original['backend_runtime_sha256'],'detector':det['runtime_sha256']},
            'scope':'owned test worker failure and recovery on same computer;not model accuracy or field trial'}
        (OUT/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
        print('PASS stopped detector→'+('automatic ownedchild recovery' if AUTOMATIC else 'normal launcher recovery')+'→real GPU task;old records retained;archive mismatch refused',flush=True)
    finally:
        if fake:fake.shutdown();fake.server_close()
        subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'stop_demo.ps1'),
            '-BackendPort',str(BACKEND_PORT),'-DetectorPort',str(DETECTOR_PORT)],capture_output=True,timeout=40,creationflags=subprocess.CREATE_NO_WINDOW)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
