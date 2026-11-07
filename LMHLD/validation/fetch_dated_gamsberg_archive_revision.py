from pathlib import Path
from datetime import datetime
from html import unescape
import json,re,sys,urllib.request
OUT=Path('D:/LMHLD/datasets/arid_accident_operation_20261004/source_review')
index=json.loads((OUT/'public_archive_index_gamsberg_reference.raw').read_text(encoding='utf-8'))
candidates=[r for r in index[1:] if r[1]>='20201218000000' and r[1]<'20220101000000']
selected=[candidates[0],candidates[-1]] if len(candidates)>1 else candidates
result={'checked_at':datetime.now().isoformat(timespec='seconds'),'early_snapshot_date_not_explicit':True,'attempts':[]}
for row in selected:
    url='https://web.archive.org/web/'+row[1]+'id_/'+row[2];attempt={'timestamp':row[1],'url':url}
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30) as response:data=response.read(3*1024*1024+1)
        if len(data)>3*1024*1024:raise ValueError('Archive too large')
        text=data.decode('utf-8',errors='replace');clean=' '.join(unescape(re.sub(r'<[^>]+>','',text)).split())
        hit=re.search(r'dated\s+3(?:rd)?\s+December',clean,re.I)
        attempt['date_statement_verified']=hit is not None
        if hit:
            path=OUT/'dated_archived_gamsberg_article.html';path.write_bytes(data)
            attempt['date_excerpt']=clean[max(0,hit.start()-60):hit.end()+50]
            result.update(date_verified=True,capture_date='2020-12-03',archive_timestamp=row[1],path=str(path))
        result['attempts'].append(attempt)
        if hit:break
    except Exception as exc:attempt['error']=str(exc);result['attempts'].append(attempt)
(OUT/'dated_archive_revision.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
