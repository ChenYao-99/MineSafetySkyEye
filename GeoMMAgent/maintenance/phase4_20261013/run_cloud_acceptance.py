"""User-approved three-task report integration; bounded, logged provider requests.

Real model/dispatcher/worker/quality calls, with previously measured GPU output
replayed into the Flask route. Inference and archival were separately verified.
Search is disabled. Only the reviewed DashScope endpoint may receive requests.
"""
from pathlib import Path
from datetime import datetime
from contextlib import ExitStack
from types import SimpleNamespace
from urllib.parse import urlparse
from unittest.mock import patch
import base64
import copy
import hashlib
import io
import json
import os
import re
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SCRATCH=OUT/'_scratch';SCRATCH.mkdir(exist_ok=True)
os.environ.update(KAT_LOCAL_FACTS_ONLY='0',KAT_CASE_ARCHIVE_PATH=str(SCRATCH/'cloud_acceptance.sqlite3'),
                  TEMP=str(SCRATCH),TMP=str(SCRATCH),YOLO_AUTOINSTALL='false')
sys.path.insert(0,str(ROOT))


def read(path):return json.loads(Path(path).read_text('utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n','utf-8')


class ProviderAudit:
    def __init__(self,manifest):
        self.manifest=manifest;self.count=0;self.total_tokens=0;self.lock=threading.Lock()
        self.current_task=None;self.current_images=set();self.records=[]
        self.path=OUT/'provider_receipts.jsonl'
        if self.path.exists():
            prior=[json.loads(line) for line in self.path.read_text('utf-8').splitlines() if line.strip()]
            self.count=max((row.get('number',0) for row in prior),default=0)
            self.total_tokens=sum((row.get('usage') or {}).get('total_tokens',0) for row in prior if row.get('event')=='finished')
            self.records=[row for row in prior if row.get('event')=='finished']

    def append(self,value):
        with self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

    def audit_message(self,body):
        from PIL import Image
        cleaned=copy.deepcopy(body);image_count=0;text_bytes=0
        for msg in cleaned.get('messages',[]):
            content=msg.get('content')
            if isinstance(content,str):text_bytes+=len(content.encode('utf-8'))
            if isinstance(content,list):
                for part in content:
                    if part.get('type')=='image_url':
                        url=part.get('image_url',{}).get('url','')
                        if not url.startswith('data:image/') or ';base64,' not in url:
                            raise RuntimeError('Remote/unreviewed image URL blocked')
                        data=base64.b64decode(url.split(',',1)[1]);im=Image.open(io.BytesIO(data)).convert('RGB')
                        pixels=hashlib.sha256(im.tobytes()).hexdigest()
                        if pixels not in self.current_images:raise RuntimeError('Image is outside reviewed task: '+pixels)
                        part['image_url']={'approved_image_bytes_sha256':hashlib.sha256(data).hexdigest(),
                                           'RGB_pixels_sha256':pixels,'size':list(im.size)}
                        image_count+=1
                    elif 'text' in part:text_bytes+=len(str(part['text']).encode('utf-8'))
        serialized=json.dumps(cleaned,ensure_ascii=False)
        if re.search(r'[CD]:[\\/]',serialized):raise RuntimeError('Local filesystem path in outgoing payload blocked')
        return cleaned,text_bytes,image_count

    def send(self,original,client,request,*args,**kwargs):
        target=urlparse(str(request.url))
        if target.scheme!='https' or target.hostname!='dashscope.aliyuncs.com' or target.path!='/compatible-mode/v1/chat/completions':
            raise RuntimeError('Unapproved destination blocked')
        body=json.loads(request.content)
        if body.get('model') not in self.manifest['models']:raise RuntimeError('Unapproved model blocked')
        cap=body.get('max_tokens') or body.get('max_completion_tokens')
        if not isinstance(cap,int) or not 0<cap<=2000:raise RuntimeError('Output cap outside approved allowance')
        try:cleaned,text_bytes,image_count=self.audit_message(body)
        except RuntimeError as exc:
            self.append(dict(event='blocked_before_send',task_id=self.current_task,reason=str(exc)))
            raise
        reserve=text_bytes+image_count*16384+cap
        with self.lock:
            if self.count>=24:raise RuntimeError('Approved 24-request cap reached')
            if self.total_tokens+reserve>200000:raise RuntimeError('Conservative provider-token budget reached')
            self.count+=1;number=self.count
            started=time.perf_counter()
            self.append(dict(event='started',number=number,task_id=self.current_task,
                at=datetime.now().astimezone().isoformat(),payload=cleaned,
                exact_body_sha256=hashlib.sha256(request.content).hexdigest(),
                conservative_token_reservation=reserve,authentication_headers_recorded=False))
        try:
            response=original(client,request,*args,**kwargs)
            response.read()
            try:raw=response.json()
            except ValueError:raw={'non_json_response':True}
            usage=raw.get('usage') or {}
            if response.status_code==200 and not isinstance(usage.get('total_tokens'),int):
                self.total_tokens=200000  # Unknown spend prevents further requests.
            else:self.total_tokens+=usage.get('total_tokens',0)
            row=dict(event='finished',number=number,task_id=self.current_task,http_status=response.status_code,
                elapsed_seconds=round(time.perf_counter()-started,4),usage=usage,
                response=raw,authentication_headers_recorded=False)
            self.append(row);self.records.append(row)
            return response
        except Exception as exc:
            self.append(dict(event='failed',number=number,task_id=self.current_task,
                elapsed_seconds=round(time.perf_counter()-started,4),error_type=type(exc).__name__))
            raise


def main(approved_sha):
    manifest_path=OUT/'outbound_review_manifest.json'
    if approved_sha!=sha(manifest_path):raise RuntimeError('Manifest differs from explicit approval')
    manifest=read(manifest_path)
    for path,digest in manifest['exact_role_prompt_sources'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Reviewed prompt source changed: '+path)
    if (OUT/'cloud_acceptance.json').exists():raise RuntimeError('Cloud receipt exists; do not overwrite')
    import backend as service
    import httpx
    from PIL import Image
    from report_grounding import check_observation_claims,check_detection_numbers
    cfg=copy.deepcopy(service._cfg)
    for name in ('SearchAgent','RetrievalAgent'):
        if name in cfg['agents']:cfg['agents'][name]['enabled']=False
    audit=ProviderAudit(manifest);results=[]
    service._llm_check_cache.update(t=0,ok=None)
    client=service.app.test_client()
    original_send=httpx.Client.send
    def blocked(*args,**kwargs):raise RuntimeError('Extra external request disabled for reviewed acceptance')
    with ExitStack() as stack:
        stack.enter_context(patch.object(service,'_cfg',cfg))
        stack.enter_context(patch('httpx.Client.send',lambda c,r,*a,**kw:audit.send(original_send,c,r,*a,**kw)))
        stack.enter_context(patch('httpx.AsyncClient.send',side_effect=blocked))
        stack.enter_context(patch('requests.sessions.Session.request',side_effect=blocked))
        stack.enter_context(patch('urllib.request.urlopen',side_effect=blocked))
        for packet in manifest['tasks']:
            audit.current_task=packet['task_id']
            image_path=OUT/packet['image_file']
            if sha(image_path)!=packet['image_sha256']:raise RuntimeError('Reviewed image changed')
            im=Image.open(image_path).convert('RGB')
            audit.current_images={hashlib.sha256(im.tobytes()).hexdigest()}
            # CAMEL serializes a PIL JPEG with its default JPEG encoder. Permit only
            # that deterministic re-encoding of the exact reviewed image, not arbitrary pictures.
            encoded_im=im
            for _ in range(3):
                encoded=io.BytesIO();encoded_im.save(encoded,format='JPEG')
                encoded_im=Image.open(io.BytesIO(encoded.getvalue())).convert('RGB')
                audit.current_images.add(hashlib.sha256(encoded_im.tobytes()).hexdigest())
            det=read(OUT/'cloud_inputs'/(packet['task_id']+'_detector.json'))
            case_id=packet['case_id']
            try:service.case_archive.get_case(case_id)
            except service.ArchiveError:
                with patch('case_archive.uuid.uuid4',return_value=SimpleNamespace(hex=case_id)):
                    service.case_archive.create_case('公开Las Cruces云端工程验收',packet['task_id'],'非专业用户评估或独立盲测')
            started=time.perf_counter();before=audit.count
            payload=dict(model_tag='mine',case_id=case_id)
            paired=packet['kind']=='paired'
            if paired:
                # The reviewed cached report uses overlay_b64; the detector transport
                # names this identical byte sequence change_overlay_b64.
                det['change_overlay_b64']=det['overlay_b64']
            payload.update(dict(image_t1='local_frozen_input',image_t2='local_frozen_input') if paired else dict(image='local_frozen_input'))
            with patch.object(service,'detector_request',return_value=det):
                response=client.post('/api/pipeline_compare' if paired else '/api/pipeline_upload',json=payload)
            body=response.get_json();errors=[]
            if response.status_code!=200:errors.append('HTTP'+str(response.status_code))
            if body.get('generation',{}).get('status')!='live':errors.append('cloud_report_not_live')
            fact_context={**service.report_facts(det),'kind':'compare' if paired else 'single','case_id':case_id}
            errors+=check_observation_claims(body.get('final_answer'),fact_context)
            errors+=check_detection_numbers(body.get('final_answer'),fact_context)
            quality=body.get('quality') or {};scope=quality.get('evidence_context') or {}
            if scope.get('case_id')!=case_id or scope.get('request_id')!=det['request_id']:errors.append('scope_mismatch')
            if body.get('report_id'):
                archived=service.case_archive.get_report(body['report_id'])
                if archived['objective_facts']!=body['objective_facts'] or archived['final_answer']!=body['final_answer']:errors.append('archive_mismatch')
                export=client.post('/api/review_record',json=dict(report_id=body['report_id'],format='json'))
                if export.status_code!=200:errors.append('export_failed')
                else:
                    exported=json.loads(base64.b64decode(export.get_json()['content_b64']))
                    if exported['automatic_analysis']['answer']!=body['final_answer']:errors.append('export_mismatch')
            # Do not duplicate image data in receipts.
            captured={k:v for k,v in body.items() if k not in ('detection','image_path')}
            captured['detection_metadata']={k:v for k,v in body.get('detection',{}).items()
                                           if not k.endswith('_b64') and k not in ('change_instances',)}
            results.append(dict(task_id=packet['task_id'],kind=packet['kind'],HTTP_status=response.status_code,
                passed=not errors,errors=errors,actual_provider_requests=audit.count-before,
                elapsed_seconds=round(time.perf_counter()-started,3),response=captured))
            print(json.dumps(dict(task=packet['task_id'],passed=not errors,errors=errors,
                provider_requests=audit.count-before,agents=[a['name'] for a in body.get('agents',[])],
                generation=body.get('generation')),ensure_ascii=False),flush=True)
    result=dict(completed_at=datetime.now().astimezone().isoformat(),manifest_sha256=approved_sha,
        approval='用户明确同意按本清单发送，最多24次；2026-10-13本聊天',
        tasks=len(results),passed=sum(t['passed'] for t in results),all_passed=all(t['passed'] for t in results),
        provider_requests=audit.count,total_provider_tokens=audit.total_tokens,
        max_requests=24,extra_search_enabled=False,production_archive_untouched=True,
        detector_inputs='frozen outputs from real GPU acceptance, not new CNN predictions during cloud stage',
        professional_geological_correctness_scored=False,independent_mine_accuracy_scored=False,rows=results)
    write(OUT/'cloud_acceptance.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--approved-manifest-sha256',required=True)
    main(parser.parse_args().approved_manifest_sha256)
