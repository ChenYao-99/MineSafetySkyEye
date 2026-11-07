"""Resume the original public thesis with bounded, verified HTTP ranges on D:."""
from pathlib import Path
from datetime import datetime
import hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import requests
OUT=ROOT/'datasets/accident_boundary_pursuit_20261007/UPC_quarry2017'
URL='https://upcommons.upc.edu/server/api/core/bitstreams/865a3e2b-e181-4d0b-bb6e-ef0ade812d9d/content'
SIZE=41755433;MD5='c640533b0b1968ae1aeb843acc6994be'


def main():
    OUT.mkdir(parents=True,exist_ok=True);partial=OUT/'TRRC1de2.part';final=OUT/'TRRC1de2.pdf'
    session=requests.Session();session.trust_env=False
    if not final.exists():
        while (partial.stat().st_size if partial.exists() else 0)<SIZE:
            start=partial.stat().st_size if partial.exists() else 0;end=min(SIZE-1,start+2*1024**2-1)
            for attempt in range(3):
                try:
                    response=session.get(URL,headers={'Range':f'bytes={start}-{end}'},timeout=90)
                    assert response.status_code==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/{SIZE}'
                    assert len(response.content)==end-start+1
                    with partial.open('ab') as stream:stream.write(response.content)
                    print('ORIGINAL_PDF_RANGE',end+1,'/',SIZE,flush=True);break
                except (requests.RequestException,AssertionError):
                    if attempt==2:raise
                    time.sleep(2)
        assert partial.stat().st_size==SIZE and hashlib.md5(partial.read_bytes()).hexdigest()==MD5
        os.replace(partial,final)
    assert hashlib.md5(final.read_bytes()).hexdigest()==MD5
    receipt={'acquired_at':datetime.now().astimezone().isoformat(),'url':URL,'bytes':SIZE,'publisher_MD5_verified':True,
        'sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'original_issued_date':'2018-07-02',
        'author':'Roger Ruiz Carulla','source_handle':'https://hdl.handle.net/2117/121044',
        'recent_search_index_date_not_event_or_publication_date':True,'role':'public research source triage,not admitted accident truth',
        'classifier_predictions':0}
    (OUT/'source_acquisition.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    import fitz
    doc=fitz.open(final);texts=[page.get_text() for page in doc]
    (OUT/'thesis_text.txt').write_text('\n'.join(texts),encoding='utf-8')
    print('ORIGINAL_PDF_VERIFIED',len(doc),'pages',flush=True)
    for i,text in enumerate(texts):
        if 'rupture of slope of quarry' in text.lower():
            for j in range(max(0,i-3),min(len(texts),i+3)):
                print('SOURCE_PAGE',j+1,texts[j][:6500],flush=True)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
