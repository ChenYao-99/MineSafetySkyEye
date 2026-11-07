"""Owned D: headless browser only; leave the local study service ready for the user."""
from pathlib import Path
import os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
import psutil,requests
OUT=ROOT/'cache/high_value_sprint_20261008';PROFILE=OUT/'edge_study_profile'


def main():
    assert not PROFILE.exists()
    assert not any(c.status=='LISTEN' and c.laddr.port==9227 for c in psutil.net_connections(kind='tcp'))
    session=requests.Session();session.trust_env=False
    assert session.get('http://127.0.0.1:5198/api/health',timeout=5).json()['service']=='kat-task-study'
    browser=None
    try:
        with (OUT/'edge_study.log').open('wb') as log:
            browser=subprocess.Popen([r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-extensions','--disable-crash-reporter',
                '--remote-debugging-port=9227','--user-data-dir='+str(PROFILE),'about:blank'],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        for _ in range(100):
            try:
                if session.get('http://127.0.0.1:9227/json/version',timeout=2).status_code==200:break
            except requests.RequestException:pass
            time.sleep(.1)
        else:raise RuntimeError('Headless study browser did not start')
        with (OUT/'browser_study.log').open('wb') as log:
            result=subprocess.run([r'C:\Program Files\nodejs\node.exe',str(ROOT/'tests/browser_task_study_smoke.mjs')],stdout=log,stderr=subprocess.STDOUT,
                timeout=90,creationflags=subprocess.CREATE_NO_WINDOW)
        assert result.returncode==0,'See browser_study.log'
        print('PASS study actual browser;real user participation remains0;local entry ready on5198',flush=True)
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
