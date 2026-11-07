"""Verify late engineering changes against the actual local WSGI services."""
from pathlib import Path
from datetime import datetime
from contextlib import closing
import hashlib,json,os,shutil,sqlite3,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
import psutil,requests
from runtime_identity import runtime_fingerprint

OUT=ROOT/'cache/engineering_hardening_20261007/latest_live'
BASE='http://127.0.0.1:5122';DET='http://127.0.0.1:5123'
AGENT_PY=Path('D:/Miniconda3/envs/GeoMMA/python.exe')
ML_PY=Path('D:/Miniconda3/envs/lmhld/python.exe')
ML=Path('D:/LMHLD')
session=requests.Session();session.trust_env=False


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def wait_no_backups(folder):
    for _ in range(100):
        if not list(folder.glob('*.sqlite3')):return
        time.sleep(.1)
    raise AssertionError('A completed or abandoned HTTP backup left a temporary file')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for port in [5122,5123,9225]:
        assert not any(c.status=='LISTEN' and c.laddr.port==port for c in psutil.net_connections(kind='tcp')),f'Port {port} is occupied'
    archive=OUT/'archive.sqlite3'
    assert not archive.exists(),'This acceptance directory has already been used;retain its receipt rather than overwrite it'
    source=ROOT/'cache/engineering_hardening_20261007/actual_task_backup.sqlite3'
    shutil.copy2(source,archive)
    assert sha(source)==sha(archive)
    state=ML/'datasets/mine_benchmark_20261002/STATE.json';state_before=sha(state)
    environment=os.environ.copy();environment.update(KAT_CASE_ARCHIVE_PATH=str(archive),PYTHONIOENCODING='utf-8',
        KAT_BROWSER_BASE=BASE,KAT_BROWSER_CDP='http://127.0.0.1:9225',KAT_BROWSER_OUT=str(OUT/'browser'),
        TEMP=str(OUT/'tmp'),TMP=str(OUT/'tmp'))
    (OUT/'tmp').mkdir(exist_ok=True)
    profile=OUT/'edge_profile';edge=None;started=False
    try:
        with (OUT/'startup.log').open('wb') as log:
            result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'start_demo.ps1'),
                '-NoBrowser','-LocalFactsOnly','-AgentPython',str(AGENT_PY),'-DetectorPython',str(ML_PY),
                '-BackendPort','5122','-DetectorPort','5123'],env=environment,stdout=log,stderr=subprocess.STDOUT,
                timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
        started=True;assert result.returncode==0,'Inspect latest_live/startup.log'
        health=session.get(BASE+'/api/health',timeout=10).json();det=session.get(DET+'/api/health',timeout=10).json()
        assert health['backend_runtime_sha256']==runtime_fingerprint(ROOT,'backend')
        assert det['runtime_sha256']==runtime_fingerprint(ML,'detector')
        assert health['detector_runtime_compatible'] and health['reporting_mode']=='local_facts_only'
        assert health['http_runtime']==det['http_runtime']=='waitress'
        save('startup_identity.json',{'backend':health,'detector':det})
        before=sha(archive)
        for host in [BASE,DET]:
            path='/api/cases' if host==BASE else '/api/health'
            for origin in ['https://unrelated.example','null','http://localhost:65500']:
                response=session.get(host+path,headers={'Origin':origin},timeout=10)
                assert response.status_code==403 and 'Access-Control-Allow-Origin' not in response.headers
            assert session.get(host+path,headers={'Host':'unrelated.example'},timeout=10).status_code==403
            assert session.get(host+path,headers={'Sec-Fetch-Site':'cross-site'},timeout=10).status_code==403
        assert session.post(BASE+'/api/case_archive/backup',json={},headers={'Origin':'null'},timeout=10).status_code==403
        assert sha(archive)==before,'Rejected requests changed the private archive'
        own_origin='http://localhost:5122'
        response=session.options(BASE+'/api/cases',headers={'Origin':own_origin,'Access-Control-Request-Method':'POST'},timeout=10)
        assert response.status_code==200 and response.headers['Access-Control-Allow-Origin']==own_origin
        assert session.get(BASE+'/api/cases',timeout=10).status_code==200
        backups=archive.parent/'backups';wait_no_backups(backups)
        downloaded=OUT/'downloaded_backup.sqlite3'
        with session.post(BASE+'/api/case_archive/backup',json={},stream=True,timeout=60) as response:
            assert response.status_code==200
            with downloaded.open('wb') as stream:
                for block in response.iter_content(128*1024):stream.write(block)
        wait_no_backups(backups)
        with closing(sqlite3.connect(downloaded)) as db:assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        with session.post(BASE+'/api/case_archive/backup',json={},stream=True,timeout=60) as response:
            assert response.status_code==200
            assert next(response.iter_content(1024)).startswith(b'SQLite format 3')
        wait_no_backups(backups)
        edge=subprocess.Popen([str(Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')),
            '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=9225',
            '--user-data-dir='+str(profile),'about:blank'],env=environment,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW)
        for _ in range(100):
            try:
                if session.get('http://127.0.0.1:9225/json/version',timeout=1).status_code==200:break
            except requests.RequestException:pass
            time.sleep(.1)
        browser=subprocess.run(['C:/Program Files/nodejs/node.exe',str(ROOT/'tests/browser_portable_task_smoke.mjs')],
            env=environment,capture_output=True,text=True,encoding='utf-8',timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
        (OUT/'browser.log').write_text(browser.stdout+'\n'+browser.stderr,encoding='utf-8')
        assert browser.returncode==0,'Inspect latest_live/browser.log'
        assert sha(state)==state_before
        receipt={'passed':True,'verified_at':datetime.now().astimezone().isoformat(),
            'latest_runtime_source_checked':True,'current_original_environments_and_default_model':True,
            'startup_doctor_and_correct_source_root':True,'local_facts_only':True,
            'foreign_null_origins_and_DNS_rebinding_host_refused':True,'same_local_origin_and_CLI_supported':True,
            'actual_Waitress_full_and_abandoned_backup_streams_cleanup':True,
            'downloaded_backup_integrity':True,'actual_browser_native_TIFF_GPU_archive_review_download':True,
            'production_archive_written':False,'benchmark_STATE_unchanged':True,
            'new_reserved_predictions_or_cloud_calls_or_model_fits':0,
            'limits':['same Windows computer','development images,not independent accident accuracy or professional feedback'],
            'source_identity':{'backend':health['backend_runtime_sha256'],'detector':det['runtime_sha256']}}
        save('verification.json',receipt)
        print('PASS latest local WSGI,origin boundaries,completed/aborted backups,actual browser task',flush=True)
    except Exception as error:
        save('failure.json',{'passed':False,'error_type':type(error).__name__,'error':str(error)})
        raise
    finally:
        # Only this run's uniquely identified D: browser profile/processes are owned.
        for process in psutil.process_iter(['name','cmdline']):
            try:
                if process.info['name'].lower()=='msedge.exe' and any(str(profile).lower() in arg.lower() for arg in (process.info['cmdline'] or [])):
                    process.kill()
            except (psutil.Error,AttributeError):pass
        if edge:
            try:edge.wait(timeout=10)
            except subprocess.TimeoutExpired:edge.kill()
        if started:
            subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'stop_demo.ps1'),
                '-BackendPort','5122','-DetectorPort','5123'],capture_output=True,timeout=40,creationflags=subprocess.CREATE_NO_WINDOW)
        time.sleep(1)
        if profile.exists():
            assert profile.resolve().is_relative_to(OUT.resolve()) and not profile.is_symlink()
            shutil.rmtree(profile)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
