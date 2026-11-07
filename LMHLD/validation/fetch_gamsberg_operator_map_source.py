from pathlib import Path
import json,sys,urllib.request,hashlib
OUT=Path('D:/LMHLD/datasets/arid_accident_operation_20261004/source_review')
url='https://vedanta-zincinternational.com/wp-content/uploads/2024/06/4-IWWMP.pdf';path=OUT/'gamsberg_operator_water_plan_2024.pdf'
if not path.exists():
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=40) as r:data=r.read(24*1024*1024+1)
    if len(data)>24*1024*1024 or not data.startswith(b'%PDF'):raise ValueError('Invalid/big operator map source')
    path.write_bytes(data)
import fitz
doc=fitz.open(path);hits=[]
for i,page in enumerate(doc):
    text=page.get_text()
    if any(w in text.lower() for w in ('latitude','longitude','18°','29°','18.','29.')) or 'Overall Water Uses' in text:
        hits.append({'pdf_page':i+1,'text':text[:10000]})
        if len(hits)<=8:page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(OUT/f'gamsberg_operator_map_page{i+1:03}.png')
p={'source_url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pages':len(doc),'coordinate_or_map_hits':hits,
   'role':'operator geographic source review, not accident GIS or new safety testimony'}
(OUT/'operator_map_inspection.json').write_bytes((json.dumps(p,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps({'pages':len(doc),'hits':[r['pdf_page'] for r in hits],'first_texts':[r['text'][:2200] for r in hits[:3]]}))
