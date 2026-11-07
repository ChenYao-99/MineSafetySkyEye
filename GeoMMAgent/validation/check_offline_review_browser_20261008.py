"""Actual local-file browser check, owning only its D: profile and process tree."""
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
import psutil,requests
OUT=ROOT/'cache/case_delivery_20261008_v2';PROFILE=OUT/'edge_offline_profile'


def main():
    assert not PROFILE.exists()
    assert not any(c.status=='LISTEN' and c.laddr.port==9228 for c in psutil.net_connections(kind='tcp'))
    browser=None;session=requests.Session();session.trust_env=False
    try:
        with (OUT/'edge_offline.log').open('wb') as log:
            browser=subprocess.Popen([r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-extensions','--disable-crash-reporter',
                '--remote-debugging-port=9228','--user-data-dir='+str(PROFILE),'about:blank'],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        for _ in range(100):
            try:
                if session.get('http://127.0.0.1:9228/json/version',timeout=2).status_code==200:break
            except requests.RequestException:pass
            time.sleep(.1)
        else:raise RuntimeError('Owned browser failed to start')
        with (OUT/'offline_browser.log').open('wb') as log:
            result=subprocess.run([r'C:\Program Files\nodejs\node.exe',str(ROOT/'tests/browser_offline_review_smoke.mjs')],stdout=log,stderr=subprocess.STDOUT,
                timeout=45,creationflags=subprocess.CREATE_NO_WINDOW)
        assert result.returncode==0,'See offline_browser.log'
        print('PASS actual browser offline handoff;single and pair;no backend or network dependency',flush=True)
    finally:
        if browser:
            try:
                owner=psutil.Process(browser.pid);children=owner.children(recursive=True)
                for process in children+[owner]:
                    try:process.terminate()
                    except psutil.NoSuchProcess:pass
                _,alive=psutil.wait_procs(children+[owner],timeout=5)
                for process in alive:
                    try:process.kill()
                    except psutil.NoSuchProcess:pass
            except psutil.NoSuchProcess:pass


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
