"""Archive traceable primary sources for arid failure and regulator-supported operations."""
from pathlib import Path
from datetime import datetime
import concurrent.futures,hashlib,json,sys,urllib.request
OUT=Path('D:/LMHLD/datasets/arid_accident_operation_20261004/source_review')
SOURCES={
 'gamsberg_operator_2021.html':'https://www.vedantalimited.com/Vedanta2021/zinc-international.html',
 'gamsberg_geotechnical_2025.pdf':'https://papers.acg.uwa.edu.au/d/2535_52_Pretorius/52_Pretorius.pdf',
 'slope_failure_review_2022.pdf':'https://mdpi-res.com/d_attachment/mining/mining-02-00019/article_deploy/mining-02-00019-with-cover.pdf?version=1690317329',
 'faxe_regulator_overburden_2022.pdf':'https://app-rsjdxp-cms-prod-001.azurewebsites.net/media/wt3k122o/13-07-2022-thornfjeldsvej-0.pdf',
 'faxe_regulator_index.html':'https://www.regionsjaelland.dk/klima-og-miljoe/raastoffer/raastoftilladelser-tilsyn-og-anmeldelser'}


def fetch(item):
    name,url=item;path=OUT/name;row={'url':url,'path':str(path),'role':'source review; no labels or model prediction'}
    try:
        if not path.exists():
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=35) as r:data=r.read(40*1024*1024+1)
            if len(data)>40*1024*1024:raise ValueError('Source exceeds40MB budget')
            if name.endswith('.pdf') and not data.startswith(b'%PDF'):raise ValueError('Expected original PDF')
            path.write_bytes(data)
        row.update(status='downloaded',bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except Exception as exc:row.update(status='unavailable',reason=str(exc))
    return name,row


OUT.mkdir(parents=True,exist_ok=True)
with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=dict(pool.map(fetch,SOURCES.items()))
p={'downloaded_at':datetime.now().isoformat(timespec='seconds'),'sources':rows,'model_predictions':0,'new_professional_project_feedback':0,
   'evidence_rules':'operator confirms event; paper or regulator permit is not automatically accepted pixel truth or proof of no hidden instability',
   'gamsberg2020_event':'2020-11-17 SouthPit','gamsberg2023_event':'2023-08-03 WestPit, same mine group',
   'gamsberg_settling_and_undercut_not_normal_negatives':True}
(OUT/'acquisition.json').write_bytes((json.dumps(p,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps({k:{'status':v['status'],'bytes':v.get('bytes'),'reason':v.get('reason')} for k,v in rows.items()},ensure_ascii=False))
