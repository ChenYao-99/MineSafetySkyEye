"""Archive and inspect the primary official accident report, without creating labels."""
from pathlib import Path
from datetime import datetime
import hashlib,json,sys,urllib.request
OUT=Path('D:/LMHLD/datasets/mine_benchmark_20261002/development/Xinjing/independent_official_20261004')
URL='https://www.mem.gov.cn/gk/zfxxgkpt/fdzdgknr/202308/W020240204579335303864.pdf'
PAGE='https://www.mem.gov.cn/gk/zfxxgkpt/fdzdgknr/202308/t20230829_460802.shtml'


def run():
    OUT.mkdir(exist_ok=True);path=OUT/'official_accident_report.pdf'
    if not path.exists():
        req=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0','Referer':PAGE})
        with urllib.request.urlopen(req,timeout=45) as r:data=r.read()
        if not data.startswith(b'%PDF'):raise ValueError('Official attachment is not PDF')
        path.write_bytes(data)
    import fitz
    doc=fitz.open(path)
    figures=[]
    for i in (6,13,14,15):
        png=OUT/f'official_page_{i+1:02}.png'
        doc[i].get_pixmap(matrix=fitz.Matrix(2,2)).save(png);figures.append(str(png))
    text='\n'.join(page.get_text() for page in doc)
    # Store numeric facts; no full extracted copyrighted text is republished.
    p={'downloaded_at':datetime.now().isoformat(timespec='seconds'),'source_page':PAGE,'attachment_url':URL,
       'publisher':'Ministry of Emergency Management, State Council investigation group',
       'pdf_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pages':len(doc),'review_images':figures,
       'role':'previously observed development event; independent official investigation source',
       'new_accident_event':False,'georeferenced_boundary_obtained':False,'new_expert_project_feedback':0,
       'numeric_extent_reported':{'north_south_max_m':630,'east_west_max_m':520,'approximate_damage_area_m2':230000},
       'numeric_extent_text_checked':all(s in text for s in ('630','520','23 万平方米')),
       'use_status':'pending visual examination; no labels derived, no model inference'}
    (OUT/'source_manifest.json').write_bytes((json.dumps(p,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'pages':len(doc),'sha256':p['pdf_sha256'],'facts_checked':p['numeric_extent_text_checked'],'images':figures}))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
