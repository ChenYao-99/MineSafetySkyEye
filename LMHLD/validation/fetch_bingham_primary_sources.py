"""Save the original field-research papers for a reserved candidate; no labels or predictions."""
from pathlib import Path
import sys,json,hashlib
import requests
ROOT=Path(__file__).resolve().parents[1]/'datasets/mine_benchmark_20261002/reserved_candidates/US_BC2021/source_review'
SOURCES={'ergun2023_back_analysis':'https://papers.acg.uwa.edu.au/d/2335_37_Ergun/37_Ergun.pdf',
         'schafer2023_rockfalls':'https://papers.acg.uwa.edu.au/d/2335_49_Schafer/49_Schafer.pdf'}


def run():
    ROOT.mkdir(parents=True,exist_ok=True);records={}
    for key,url in SOURCES.items():
        target=ROOT/(key+'.pdf');entry={'url':url,'role':'reserved_source_review_no_predictions'}
        try:
            if not target.exists():
                with requests.get(url,stream=True,timeout=(10,35)) as response:
                    response.raise_for_status();data=bytearray()
                    for chunk in response.iter_content(1024*1024):
                        data.extend(chunk)
                        if len(data)>30*1024*1024:raise ValueError('Primary PDF size cap exceeded')
                if not data.startswith(b'%PDF'):raise ValueError('Not PDF')
                target.write_bytes(data)
            entry.update(status='downloaded',bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            print(key,entry['bytes'],'bytes',flush=True)
        except Exception as exc:entry.update(status='unavailable',reason=str(exc))
        records[key]=entry
    (ROOT/'acquisition.json').write_text(json.dumps({'sources':records,'model_predictions':0,'original_expert_GIS_obtained':False},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
