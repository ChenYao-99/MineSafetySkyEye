"""One known-development inference, then archive/review/export without any replay."""
from pathlib import Path
import base64,hashlib,json,sys,time,urllib.request
ROOT=Path('D:/GeoMMAgent');sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
OUT=ROOT/'cache/archive_workflow_acceptance_20261006';OUT.mkdir(exist_ok=True)
BASE='http://127.0.0.1:5105'


def post(path,data):
    request=urllib.request.Request(BASE+path,data=json.dumps(data).encode('utf-8'),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=180) as response:return json.loads(response.read())


def main():
    image=Path('D:/LMHLD/datasets/mine_benchmark_20261002/development/LasCruces/post_BGRN_10m.tif')
    case=post('/api/cases',{'mine_name':'LasCruces 开发影像工程验收','title':'实际模型与归档闭环','notes':'已观察开发数据，不是新盲测或现场试用'})
    start=time.monotonic()
    result=post('/api/pipeline_upload',{'image':base64.b64encode(image.read_bytes()).decode(),'model_tag':'mine','case_id':case['case_id']})
    elapsed=time.monotonic()-start
    assert result['generation']['status']=='fallback' and result['archive']['persisted']
    assert result['case_context']['case_id']==case['case_id']
    assert result['review_handoff']['trace']['trace_complete']
    assert result['review_handoff']['input_evidence']['image']['input_sha256']==hashlib.sha256(image.read_bytes()).hexdigest()
    assert result['review_handoff']['trace']['weights_sha256']==['0d8d8a0540419d4409951ed91e69499fac206b68303f3feec08a32f0841cebef']
    reviewed=post('/api/archive/reports/'+result['report_id']+'/reviews',{
        'expected_revision':0,'target_key':'report','status':'insufficient_evidence','reviewer':'内部工程验收',
        'evidence':'原始开发影像和同次模型输出；未取得独立现场证据','note':'仅验证流程，不判断事故或安全状态'})
    assert reviewed['detection']==result['detection'] and reviewed['objective_facts']==result['objective_facts']
    record=post('/api/review_record',{'report_id':result['report_id']})
    assert record['record']['review_revision']==1
    png=post('/api/report_upload',{'report_id':result['report_id']})
    raw=base64.b64decode(png['report_b64']);assert raw.startswith(b'\x89PNG\r\n\x1a\n')
    (OUT/'real_model_reviewed_report.png').write_bytes(raw)
    receipt={'passed':True,'case_id':case['case_id'],'report_id':result['report_id'],
      'actual_default_model_inferences':1,'additional_inferences_during_review_or_export':0,
      'image_source':str(image),'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
      'request_id':result['detection']['request_id'],'weights_sha256':result['review_handoff']['trace']['weights_sha256'],
      'model_facts_unchanged_by_review':True,'review_revision':1,'pipeline_elapsed_seconds':round(elapsed,3),
      'generation':'local_grounded_fallback','new_cloud_calls':0,'new_blind_evaluation':False,
      'scope':'one observed-development functional run,not a precision/recall benchmark or long-duration reliability qualification'}
    (OUT/'real_e2e_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS actual default GPU model -> measured fallback report -> persistent archive -> review -> PNG/JSON exports; no extra inference/cloud')
    post('/api/local_e2e_shutdown',{})


if __name__=='__main__':main()
