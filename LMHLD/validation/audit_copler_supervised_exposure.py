"""Audit documented historical supervised protocols, without claiming universal pretraining knowledge."""
from pathlib import Path
from datetime import datetime
import json,re,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
CASE=ROOT/'datasets/mine_benchmark_20261002/reserved_candidates/TR_COP2024'
context=json.loads((CASE/'context.json').read_text(encoding='utf-8'))
registered=datetime.fromisoformat(context['registered_at']).timestamp()
pattern=re.compile(r'(?i)\b(?:TR_COP2024|TR_COPLER|copler|çöpler|çopler|iliç)\b')
rows=[];hits=[]
for base in (ROOT/'experiments',ROOT/'models_prior'):
 for path in base.rglob('*.json'):
  if not any(word in path.name.lower() for word in ('protocol','development_data','train','manifest')):continue
  stat=path.stat()
  if stat.st_size>2*1024*1024 or stat.st_mtime>registered:continue
  text=path.read_text(encoding='utf-8-sig',errors='replace');match=pattern.search(text)
  rows.append({'path':str(path),'sha256':file_sha256(path),'bytes':stat.st_size,'matching_explicit_new_mine_alias':bool(match)})
  if match:hits.append(str(path))
prior=ROOT/'datasets/mine_benchmark_20261002/training_provenance_audit.json'
v={'created_at':datetime.now().isoformat(timespec='seconds'),'case_id':'TR_COP2024','canonical_role':'reserved_final_test','pre_registration_supervised_protocol_files_checked':len(rows),'explicit_new_mine_matches':hits,
  'documented_known_supervised_exposure_found':bool(hits),'scope':'documented historical training/input manifest/protocol text before case reservation; no detector outputs',
  'prior_supervised_source_audit_sha256':file_sha256(prior),'generic_or_opaque_pretraining_exposure_proven_absent':False,
  'case_reserved_before_new_data_or_labels_acquired':True,'new_accident_detector_predictions':0,'files':rows}
(CASE/'documented_supervised_exposure_audit.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps({k:v[k] for k in ('pre_registration_supervised_protocol_files_checked','explicit_new_mine_matches','documented_known_supervised_exposure_found')},ensure_ascii=False))
if hits:raise SystemExit('Exposure found; keep reserved quality pending, no unseen-mine claim')
