"""Find public operational mapping products by event; do not replace reference labels automatically."""
from pathlib import Path
from datetime import datetime
import json,sys,time
import requests
OUT=Path(__file__).resolve().parents[1]/'datasets/operational_boundary_sources_20261004'
BASE='https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations-info/'


def run():
    OUT.mkdir(exist_ok=True);session=requests.Session();url=BASE+'?limit=100';seen=set();rows=[];errors=[]
    for page in range(30):
        if url in seen:raise ValueError('Repeated public catalogue URL')
        seen.add(url);path=OUT/f'rapid_mapping_page{page:02d}.json'
        try:
            if not path.exists():
                response=session.get(url,timeout=(10,25));response.raise_for_status()
                if len(response.content)>20*1024*1024:raise ValueError('Metadata unexpectedly large')
                value=response.json();path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            value=json.loads(path.read_text(encoding='utf-8'));rows.extend(value.get('results',value if isinstance(value,list) else []))
            url=value.get('next') if isinstance(value,dict) else None
            print('Operational catalogue page',page,'records',len(rows),flush=True)
            if not url:break
        except Exception as exc:
            errors.append({'url':url,'reason':str(exc)});break
    keywords=('brumad','anarg','amynt','mine','mining','dam burst','greece','landslide')
    selected=[]
    for row in rows:
        text=' '.join(str(row.get(k,'')) for k in ('name','reason','category','subCategory')).lower()
        if any(word in text for word in keywords):selected.append(row)
    record={'checked_at':datetime.now().isoformat(timespec='seconds'),'source_documentation':'https://mapping.emergency.copernicus.eu/about/how-to-harvest-cems-mapping-data/emergency-response-data/',
            'total_catalogue_records_read':len(rows),'event_candidates':selected,'errors':errors,
            'downloaded_boundary_vectors':0,'labels_changed':False,'model_predictions':0}
    (OUT/'catalogue_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'read':len(rows),'candidates':[{'code':r.get('code'),'name':r.get('name'),'eventTime':r.get('eventTime')} for r in selected],'errors':errors}))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
