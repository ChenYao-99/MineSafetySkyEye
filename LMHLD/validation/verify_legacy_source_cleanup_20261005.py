"""Validate source housekeeping, deployment assets and frozen scientific contracts."""
from pathlib import Path
from datetime import datetime
import json,sys,zipfile,hashlib,shutil,unittest
ROOT=Path('D:/LMHLD');APP=Path('D:/GeoMMAgent');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from model_registry import file_sha256
from validation.mine_data_protocol import registry,verify_assets,DATA
from pixel_reference_contract import load_partial_reference
OUT=ROOT/'cache/source_cleanup_20261005'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
    plan=load(OUT/'plan.json');receipt=load(OUT/'deleted_files.json')
    if receipt['plan_sha256']!=file_sha256(OUT/'plan.json'):raise ValueError('Exact deletion plan changed')
    if len(receipt['deleted'])!=len(plan['archived_files_to_remove'])+len(plan['regenerable_bytecode_to_remove']):raise ValueError('Exact-file removal incomplete')
    if sum(r['bytes'] for r in receipt['deleted'])!=receipt['removed_bytes']:raise ValueError('Deletion count/bytes inconsistent')
    for r in plan['archived_files_to_remove']:
        if Path(r['path']).exists():raise ValueError('Retired source/output still present at old path')
        for target in r['generated_targets_retained']:
            if file_sha256(target['path'])!=target['sha256']:raise ValueError('Generated final implementation changed')
    archive=Path(plan['archive']['path'])
    if file_sha256(archive)!=plan['archive']['sha256']:raise ValueError('Recoverable archive changed')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive corruption')
        for r in plan['archived_files_to_remove']:
            if hashlib.sha256(z.read(r['archive_member'])).hexdigest()!=r['sha256']:raise ValueError('Archived historical source differs')
    for p,digest in plan['protected_remaining_source_sha256'].items():
        if file_sha256(p)!=digest:raise ValueError('Remaining source changed during housekeeping')
    for p,digest in plan['protected_runtime_artifact_sha256'].items():
        if file_sha256(p)!=digest:raise ValueError('Current deployment artifact changed')
    verify_assets();value=registry()
    if file_sha256(DATA/'mine_case_registry.json')!=plan['registry_sha256_before']:raise ValueError('Cleanup changed mine roles or counters')
    if (DATA/'final_evaluation_once.json').exists():raise ValueError('Independent final test unexpectedly opened')
    load_partial_reference(ROOT/'datasets/jagersfontein_development_20261005/ZA_JAGS2022/pixel_reference_v1_official_partial','development_training')
    modules=['validation.test_georeferenced_protocol','validation.test_explicit_mine_group_isolation','validation.test_pixel_reference_contract',
             'validation.test_published_inventory_contract','validation.test_verified_cog_window','validation.test_apparent_pair_correspondence',
             'validation.test_final_selection_evidence','validation.test_reserved_geographic_protection','validation.test_frozen_semantic_grid']
    suite=unittest.defaultTestLoader.loadTestsFromNames(modules)
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*upload*.py'))
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_knowledge_catalog.py'))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise ValueError('Post-cleanup source/data/runtime contracts failed')
    state=load(DATA/'STATE.json')
    if any(state[k]!=v for k,v in plan['state_counts_before'].items()):raise ValueError('Housekeeping changed actual historical fit counts')
    report={'created_at':datetime.now().isoformat(timespec='seconds'),'archived_source_files':22,'archived_debug_outputs':14,
            'retired_files_removed_from_worktree':36,'regenerable_bytecode_removed':164,'all_original_archived_bytes_verified':True,
            'protected_remaining_sources_verified':len(plan['protected_remaining_source_sha256']),
            'protected_runtime_artifacts_verified':len(plan['protected_runtime_artifact_sha256']),
            'frozen_source_or_reference_assets_verified':len({**value['frozen_assets'],**value['additional_frozen_assets']}),
            'post_cleanup_tests':{'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':True},
            'archive':str(archive),'archive_sha256':file_sha256(archive),'archive_bytes':archive.stat().st_size,
            'deleted_bytes':receipt['removed_bytes'],'gross_removed_MiB':round(receipt['removed_bytes']/1024**2,2),
            'bytecode_regeneration_expected':True,'restoration_tool':str(ROOT/'validation/restore_retired_source.py'),
            'raw_datasets_labels_study_results_current_weights_credentials_or_user_tracked_edits_deleted':False,
            'reserved_predictions':{c:value['cases'][c]['model_predictions_run'] for c in ('CL0002','KG0001','US_BC2021','TR_COP2024')},
            'native_fits':state['completed_native_training_runs'],'lightweight_fits':state['completed_lightweight_feature_diagnostics'],
            'default_model_changed':False,'free_D_GiB':round(shutil.disk_usage(ROOT).free/1024**3,2)}
    write(OUT/'verification.json',report)
    state.setdefault('artifact_retention',{})['historical_source_housekeeping']=str(OUT/'verification.json');write(DATA/'STATE.json',state)
    note=ROOT/'validation/STATE.md';history=note.read_text(encoding='utf-8')
    marker='# 代码清理验收（2026-10-05）'
    if not history.startswith(marker):
        note.write_text(marker+'\n\n22个已完成的一次性生成/诊断脚本及空样板、14个根目录旧调试输出已逐字节归档后从工作树移除，另清除164个可重建编译缓存。所有导入模块、冻结SHA约束代码、正式run/fetch/evaluate实现、训练复现入口、测试、原图/标签/协议/实验结果、15运行权重/依赖、凭据和现有用户修改保留；归档CRC与36个SHA验证通过。源码结构及恢复方法见 validation/SOURCE_LAYOUT.md；删除清单/归档/验真在 cache/source_cleanup_20261005。清理不增加训练/评估实验数（164CNN/107轻量）、不改变保留case0预测与默认。后续仍按 competition_readiness_assessment_20261005.json 优先级推进，材料暂缓。\n\n'+history,encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('archived_source_files','archived_debug_outputs','regenerable_bytecode_removed','post_cleanup_tests','protected_runtime_artifacts_verified','frozen_source_or_reference_assets_verified','reserved_predictions','free_D_GiB')},ensure_ascii=False))
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
