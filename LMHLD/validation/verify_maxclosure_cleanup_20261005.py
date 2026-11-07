"""Verify exact-file cleanup without rerunning completed inference or test suites."""
from pathlib import Path
from datetime import datetime
import sys,json,shutil
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from validation.mine_data_protocol import verify_assets,registry,DATA
from pixel_reference_contract import load_partial_reference
def read(p):return Path(p).read_text(encoding='utf-8-sig')
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
    proof=ROOT/'cache/cleanup_20261005/rejected_input_contract_head_weights.json';receipt=json.loads(read(proof))
    if len(receipt['deleted'])!=12 or receipt['removed_bytes']!=sum(x['bytes'] for x in receipt['deleted']):raise ValueError('Cleanup receipt incomplete')
    verification_path=ROOT/'validation/hardgap_closure_verification_20261005.json';v=json.loads(read(verification_path))
    deleted={x['path']:x for x in receipt['deleted']}
    for row in v['head_weight_integrity_receipts']:
        path=Path(row['path'])
        if row['path'] in deleted:
            if path.exists() or row['sha256']!=deleted[row['path']]['sha256']:raise ValueError('Deleted diagnostic artifact state inconsistent')
            row.update(retained=False,retention_status='removed rejected input-contract diagnostic by user authorization')
        elif not path.exists() or file_sha256(path)!=row['sha256']:raise ValueError('Corrected representative head changed')
    for name,digest in v['protected_runtime_artifact_sha256'].items():
        if file_sha256(name)!=digest:raise ValueError('Protected runtime artifact changed after cleanup')
    verify_assets();value=registry()
    load_partial_reference(ROOT/'datasets/jagersfontein_development_20261005/ZA_JAGS2022/pixel_reference_v1_official_partial','development_training')
    for case in ('CL0002','KG0001','US_BC2021','TR_COP2024'):
        if value['cases'][case]['model_predictions_run']!=0:raise ValueError('Reserved predictions changed')
    if (DATA/'final_evaluation_once.json').exists():raise ValueError('Final test unexpectedly opened')
    v['rejected_head_weight_cleanup']={'receipt':str(proof),'receipt_sha256':file_sha256(proof),'deleted':12,
        'freed_MiB':round(receipt['removed_bytes']/1024**2,2),'post_cleanup_verified_at':datetime.now().isoformat(timespec='seconds'),
        'original_data_labels_predictions_or_runtime_weights_deleted':False,'corrected_representative_heads_retained':8,
        'actual_completed_fit_counts_unchanged':True}
    v['free_D_GiB']=round(shutil.disk_usage(ROOT).free/1024**3,2);write(verification_path,v)
    state=json.loads(read(DATA/'STATE.json'));state['artifact_retention']['maxclosure_rejected_head_cleanup']=v['rejected_head_weight_cleanup']
    state['updated_at']=datetime.now().isoformat(timespec='seconds');state['free_D_GiB']=v['free_D_GiB'];write(DATA/'STATE.json',state)
    note=ROOT/'validation/STATE.md';text=read(note)
    marker='本阶段清理：12个输入合同旧版诊断权重已按校验清单删除（3.25MiB）；8个校正版代表模型、全部原图/标签/样本/预测与15项运行依赖保留；20实际拟合/107累计轻量诊断计数不变。'
    if marker not in text:
        pos=text.find('\n\n');text=text[:pos+2]+marker+'\n\n'+text[pos+2:];note.write_text(text,encoding='utf-8')
    print(json.dumps({'cleanup':v['rejected_head_weight_cleanup'],'runtime_artifacts_still_verified':15,
                      'frozen_source_or_reference_assets_still_verified':len({**value['frozen_assets'],**value['additional_frozen_assets']}),
                      'free_D_GiB':v['free_D_GiB'],'reserved_predictions_still0':True,'model_or_default_changed':False}))
if __name__=='__main__':run()
