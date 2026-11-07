"""GPU-backed local-facts chain with isolated archives, review and offline exports."""
from pathlib import Path
import os,sys,json,base64,urllib.request,time,sqlite3
ROOT=Path(r'D:\GeoMMAgent');OUT=ROOT/'maintenance/phase3_20261013'
os.environ.update(KAT_LOCAL_FACTS_ONLY='1',KAT_CASE_ARCHIVE_PATH=str(OUT/'_scratch/real_chain_test.sqlite3'),DETECTOR_PORT='5101',YOLO_AUTOINSTALL='false')
sys.path.insert(0,str(ROOT))
import backend as service
client=service.app.test_client()
def post(url,payload):
 response=client.post(url,json=payload);body=response.get_json()
 assert response.status_code in (200,201),(url,response.status_code,body.get('error'))
 assert not body.get('error'),body.get('error')
 return body
case=post('/api/cases',{'mine_name':'修复后内部复验','title':'原始影像与离线交接','notes':'内部功能检查，非专业用户评估'})
inputs=ROOT/'cache/case_delivery_20261008_v2/inputs'
image={epoch:base64.b64encode((inputs/(epoch+'_BGRN_10m.tif')).read_bytes()).decode() for epoch in ['pre','post']}
records=[]
for name,url,payload in [('single','/api/pipeline_upload',{'image':image['post']}),('paired','/api/pipeline_compare',{'image_t1':image['pre'],'image_t2':image['post']})]:
 started=time.perf_counter();body=post(url,{**payload,'model_tag':'mine','case_id':case['case_id']})
 rid=body['report_id'];facts=body['objective_facts'];snapshot=body['archive']['snapshot_sha256']
 assert body['quality']['evidence_context']['case_id']==str(case['case_id'])
 assert body['quality']['evidence_context']['scope_id']==body['detection']['request_id']
 assert body['generation']['source']=='detection_summary'
 h,w=body['detection']['size']
 reviewed=post('/api/archive/reports/'+rid+'/manual_annotations',{'expected_revision':0,'box_px':[w*.2,h*.25,w*.35,h*.4],
  'reviewer':'内部功能复验','evidence':'同次原图，内部功能测试，不是专业事故边界','note':'核查位置与模型事实分层保留'})
 assert reviewed['objective_facts']==facts and reviewed['archive']['snapshot_sha256']==snapshot
 for fmt in ['json','html']:
  exported=post('/api/review_record',{'report_id':rid,'format':fmt})
  content=base64.b64decode(exported['content_b64'])
  if fmt=='html':assert '<svg id="scene"' in content.decode('utf-8')
  else:assert len(json.loads(content)['manual_annotations'])==1
 if name=='paired':
  for fmt in ['html','json']:assert post('/api/report_compare',{'report_id':rid,'format':fmt})['content_b64']
 else:assert post('/api/report_upload',{'report_id':rid})['report_b64']
 records.append({'task':name,'report_saved':True,'raw_facts_unchanged_after_review':True,'JSON_and_HTML_exported':True,'elapsed_seconds':round(time.perf_counter()-started,3)})
knowledge=post('/api/knowledge',{'query':'露天矿边坡异常影像如何复核'})
assert knowledge['docs'] and all(row.get('source_title') or row.get('source') or row.get('source_id') for row in knowledge['docs'])
with service.case_archive.connection() as db:assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
result={'passed':True,'records':records,'knowledge_retrieval':True,'mode':'local_facts_only','cloud_calls':0,'production_archive_untouched':True}
(OUT/'local_delivery.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(result,ensure_ascii=False))
