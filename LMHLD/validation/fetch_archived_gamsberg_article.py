from pathlib import Path
from datetime import datetime
import json,re,sys,urllib.request
OUT=Path('D:/LMHLD/datasets/arid_accident_operation_20261004/source_review')
records=json.loads((OUT/'public_archive_index_gamsberg_reference.raw').read_text(encoding='utf-8'))
snapshot=next(r for r in records[1:] if r[1].startswith('202012'))
url='https://web.archive.org/web/'+snapshot[1]+'id_/'+snapshot[2]
p={'archive_snapshot':snapshot[1],'original_url':snapshot[2],'retrieval_url':url,'model_predictions':0}
try:
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=35) as r:data=r.read(3*1024*1024+1)
    if len(data)>3*1024*1024:raise ValueError('Archived page too large')
    path=OUT/'original_archived_gamsberg_article.html';path.write_bytes(data);text=data.decode('utf-8',errors='replace')
    matches=[]
    for img in re.findall(r'<img[^>]+>',text):
        if 'gamsberg' in img.lower():matches.append(img)
    p.update(path=str(path),image_tags=matches,original_article_content_verified='3rd December' in text,
             role='public archived primary reference; actual imagery licence unchanged')
except Exception as exc:p['error']=str(exc)
(OUT/'archived_article_acquisition.json').write_bytes((json.dumps(p,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps(p,ensure_ascii=False))
