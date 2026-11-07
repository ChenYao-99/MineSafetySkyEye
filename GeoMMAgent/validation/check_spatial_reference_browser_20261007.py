"""Owned local services and an actual browser task; controlled development TIFF only."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
import psutil,requests

OUT=ROOT/'cache/spatial_reference_acceptance_20261007'
UI=OUT/'browser_reference'
BASE='http://127.0.0.1:5190'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1)
    attempt=parser.parse_args().attempt
    assert 1<=attempt<=9
    suffix='' if attempt==1 else '_v'+str(attempt)
    UI=OUT/('browser_reference'+suffix)
    UI.mkdir(exist_ok=True)
    archive=OUT/('archive_ui'+suffix+'.sqlite3')
    profile=OUT/('edge_scale_profile'+suffix)
    assert not archive.exists(),'Preserve an existing acceptance archive'
    assert not profile.exists(),'Never reuse or clean an existing browser profile'
    assert not any(c.status=='LISTEN' and c.laddr.port in [5190,5191,9226] for c in psutil.net_connections(kind='tcp'))
    image=OUT/'LasCruces_controlled_2p5m.tif'
    assert image.is_file()
    env=os.environ.copy()
    env.update(KAT_CASE_ARCHIVE_PATH=str(archive),PYTHONIOENCODING='utf-8',
        KAT_BROWSER_BASE=BASE,KAT_BROWSER_OUT=str(UI),KAT_BROWSER_CDP='http://127.0.0.1:9226',
        KAT_BROWSER_INPUT=str(image),KAT_BROWSER_GRID='reference_10m',KAT_BROWSER_EXPECTED_GSD='2.5')
    browser=None
    session=requests.Session();session.trust_env=False
    try:
        with (UI/'startup.log').open('wb') as log:
            result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'start_demo.ps1'),
                '-NoBrowser','-LocalFactsOnly','-BackendPort','5190','-DetectorPort','5191'],
                env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
        assert result.returncode==0,'See startup.log'
        health=session.get(BASE+'/api/health',timeout=10).json()
        detector=session.get('http://127.0.0.1:5191/api/health',timeout=10).json()
        assert health['detector_runtime_compatible'] and health['reporting_mode']=='local_facts_only'
        with (UI/'edge.log').open('wb') as log:
            browser=subprocess.Popen([r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
                '--disable-extensions','--disable-crash-reporter','--remote-debugging-port=9226',
                '--user-data-dir='+str(profile),'about:blank'],env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
        for _ in range(100):
            try:
                if session.get('http://127.0.0.1:9226/json/version',timeout=2).status_code==200:break
            except requests.RequestException:pass
            time.sleep(.2)
        else:raise RuntimeError('Owned headless browser failed to start')
        with (UI/'browser_task.log').open('wb') as log:
            result=subprocess.run([r'C:\Program Files\nodejs\node.exe',str(ROOT/'tests/browser_portable_task_smoke.mjs')],
                env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
        assert result.returncode==0,'See browser_task.log'
        receipt=json.loads((UI/'verification.json').read_text(encoding='utf-8'))
        assert receipt['passed'] and receipt['analysis_grid']=='reference_10m'
        receipt['runtime_identity']={'backend':health['backend_runtime_sha256'],'detector':detector['runtime_sha256']}
        receipt['production_archive_written']=False
        receipt['controlled_replica_is_not_new_independent_sensor_imagery']=True
        (UI/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('PASS owned actual browser: reference scale upload, GPU, report, archive and both vector downloads',flush=True)
    finally:
        if browser:
            try:
                owned=psutil.Process(browser.pid)
                children=owned.children(recursive=True)
                for child in children:
                    try:child.terminate()
                    except psutil.NoSuchProcess:pass
                owned.terminate()
                _,alive=psutil.wait_procs(children+[owned],timeout=5)
                for process in alive:
                    try:process.kill()
                    except psutil.NoSuchProcess:pass
            except psutil.NoSuchProcess:pass
        subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'stop_demo.ps1'),
            '-BackendPort','5190','-DetectorPort','5191'],env=env,capture_output=True,timeout=40,creationflags=subprocess.CREATE_NO_WINDOW)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
