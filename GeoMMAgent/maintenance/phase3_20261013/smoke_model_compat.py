"""Real provider compatibility via the Flask QA route, using only a synthetic shape.

The test supplies no mine imagery, identity, source code, dataset facts, or case
database to the provider. It does not measure mine recognition or warning quality.
"""
from pathlib import Path
import base64
import copy
import hashlib
import io
import json
import os
import sys
import time

ROOT = Path(r'D:\GeoMMAgent')
OUT = ROOT/'maintenance/phase3_20261013'
SCRATCH = OUT/'_scratch'
SCRATCH.mkdir(exist_ok=True)
os.environ['KAT_CASE_ARCHIVE_PATH'] = str(SCRATCH/'compatibility.sqlite3')
os.environ['TEMP'] = str(SCRATCH)
os.environ['TMP'] = str(SCRATCH)
sys.path.insert(0,str(ROOT))
from PIL import Image, ImageDraw
import backend as service

config = copy.deepcopy(service._cfg)
for name, value in config['agents'].items():
    value['enabled'] = name == 'ReasoningAgent'
service._cfg = config
service._llm_check_cache.update(t=0,ok=None)
image = Image.new('RGB',(64,64),'white')
ImageDraw.Draw(image).rectangle((16,20,48,44),fill='red')
buffer=io.BytesIO();image.save(buffer,format='PNG')
question='这是合成接口测试图，请只描述中间图形的颜色与形状，不做地质或风险判断，不输出选项字母。'
started=time.perf_counter()
response=service.app.test_client().post('/api/analyze',json={
    'question':question,'image':base64.b64encode(buffer.getvalue()).decode()})
body=response.get_json()
trace=None
if body.get('trace_dir'):
    trace_files=list(Path(body['trace_dir']).glob('*.json'))
    # Preserve the full trace plus per-step records in one small test receipt.
    trace={p.name:json.loads(p.read_text(encoding='utf-8')) for p in trace_files}
answer=body.get('final_answer','')
passed=(response.status_code==200 and not body.get('is_mock')
        and '红' in answer and any(word in answer for word in ['矩形','方形','长方形','方块'])
        and len(body.get('agents',[]))>=1)
record={'scope':'actual provider and Flask QA route compatibility; synthetic image only',
    'passed':passed,'HTTP_status':response.status_code,'elapsed_s':time.perf_counter()-started,
    'model':service._cfg['coordinator']['model'],
    'fixture':{'size':[64,64],'background':'white','rectangle':[16,20,48,44],'fill':'red',
               'PNG_sha256':hashlib.sha256(buffer.getvalue()).hexdigest()},
    'question':question,'response':body,'trace':trace,
    'professional_or_mine_accuracy_evaluated':False,
    'mine_images_or_private_evidence_uploaded':False,
    'production_archive_written':False,
    'cleanup_paths':[str(SCRATCH)],
    'cleanup_files':[body['image_path']] if body.get('image_path') else [],
    'cleanup_directories':[body['trace_dir']] if body.get('trace_dir') else []}
(OUT/'cloud_compatibility.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:record[k] for k in ['passed','HTTP_status','elapsed_s','model']},ensure_ascii=False))
print('REAL_ANSWER',answer)
sys.exit(0 if passed else 1)
