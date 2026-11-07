"""Locate public slope-geometry data; labels cannot be reused as accident or safety labels."""
from pathlib import Path
import json,sys
from datetime import datetime
import requests
OUT=Path(__file__).resolve().parents[1]/'datasets/slope_geometry_auxiliary_20261004'


def run():
    OUT.mkdir(exist_ok=True);session=requests.Session();records=[]
    response=session.get('https://api.figshare.com/v2/articles/31674361',timeout=(10,30))
    published_status=response.status_code
    if response.ok:
        published=response.json()
        (OUT/'article_31674361.json').write_text(json.dumps(published,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        records.append(published)
    for query in ('OPMS-Seg','open pit coal mine slopes'):
        response=session.post('https://api.figshare.com/v2/articles/search',json={'search_for':query,'limit':30},timeout=(10,30))
        response.raise_for_status();values=response.json()
        (OUT/('search_'+query.replace(' ','_')+'.json')).write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for value in values:
            title=value.get('title','')
            if 'OPMS' in title.upper():
                records.append(value)
    seen=set();details=[]
    for value in records:
        if value['id'] in seen:continue
        seen.add(value['id']);response=session.get(f'https://api.figshare.com/v2/articles/{value["id"]}',timeout=(10,30));response.raise_for_status()
        metadata=response.json();details.append(metadata)
        (OUT/f'article_{value["id"]}.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'id':value['id'],'title':metadata.get('title'),'files':[{'name':f['name'],'bytes':f['size']} for f in metadata.get('files',[])]}),flush=True)
    plan={'checked_at':datetime.now().isoformat(timespec='seconds'),'articles_found':[v['id'] for v in details],
          'published_DOI':'10.6084/m9.figshare.31674361','published_link_HTTP_status':published_status,
          'role':'auxiliary visible slope geometry only','accident_or_stability_labels':False,
          'no_new_expert_feedback_for_this_project':True,'download_started':False,'training_started':False,
          'required_before_use':['mine provenance and reserved-mine exclusion','actual polygon/image integrity','group by mine/flight/source frame, not arbitrary crops','camera oblique view and unknown GSD cannot be called native10m satellite data']}
    (OUT/'source_role.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'matched_articles':len(details),'downloaded_raw_images':0}))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
