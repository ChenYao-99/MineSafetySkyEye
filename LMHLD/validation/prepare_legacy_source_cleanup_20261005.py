"""Prepare exact-file source housekeeping with a verified recoverable archive."""
from pathlib import Path
from datetime import datetime
import json,sys,os,ast,zipfile,hashlib,re
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from validation.mine_data_protocol import registry,verify_assets,DATA
ROOTS=[Path('D:/GeoMMAgent'),ROOT];OUT=ROOT/'cache/source_cleanup_20261005'
AUDIT=OUT/'audit.json'
GENERATORS={
 'make_arid_weak_presence_contrasts.py':['run_arid_weak_presence_operation_20261004.py'],
 'make_direct_winter_regularizer_study.py':['run_direct_winter_regularizer_study.py'],
 'make_diverse_material_contrasts.py':['run_diverse_material_data_20261004.py','run_quality_diverse_material_20261004.py'],
 'make_diverse_operation_fetcher.py':['fetch_diverse_operation_pairs.py'],
 'make_diverse_quality_fallback_fetcher.py':['fetch_diverse_operation_quality_fallbacks.py'],
 'make_four_mine_calibrated_probe.py':['four_mine_calibrated_probe.py'],
 'make_matched_arid_operation_contrast.py':['run_arid_weak_presence_operation_matched_20261004.py'],
 'make_normal_focal_runner.py':['run_normal_focal_study.py'],
 'make_reviewed_normal_evaluator.py':['evaluate_reviewed_normal_study.py'],
 'make_reviewed_normal_runner.py':['run_reviewed_normal_study.py'],
 'make_rgb_colour_style_pilot.py':['run_rgb_colour_style_pilot.py'],
 'make_rgb_direct_context_study.py':['run_rgb_direct_context_study.py'],
 'make_rgb_soft_prior_pilot.py':['run_rgb_soft_prior_pilot.py'],
 'make_winter_normal_study.py':['run_winter_normal_study.py']}
INSPECTORS=['inspect_arid_source_figures.py','inspect_brazil_source_links.py','inspect_copler_source_structure.py','inspect_jagersfontein_red_components.py',
            'probe_charter_public_products_20261005.py','probe_public_gamsberg_reference_access.py']
SOURCE_SUFFIX={'.py','.js','.mjs','.bat','.sh','.ps1','.yaml','.yml'}
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def root_for(p):
    p=p.resolve()
    for root in ROOTS:
        if p.is_relative_to(root.resolve()):return root
    raise ValueError('Source housekeeping must stay inside authorised D: workspaces')
def non_reparse_file(p):
    root=root_for(p);cur=p.resolve()
    while cur!=root.parent:
        if cur.stat().st_file_attributes & 0x400:raise ValueError('Reparse ancestor is not a safe cleanup target')
        cur=cur.parent
    if not p.is_file():raise ValueError('Exact ordinary files required')
def record(p,kind,reason,targets=()):
    non_reparse_file(p);root=root_for(p)
    return {'path':str(p),'root':str(root),'relative':p.relative_to(root).as_posix(),'archive_member':root.name+'/'+p.relative_to(root).as_posix(),
            'kind':kind,'reason':reason,'bytes':p.stat().st_size,'sha256':file_sha256(p),
            'generated_targets_retained':[{'path':str(t),'sha256':file_sha256(t)} for t in targets]}
def run():
    verify_assets();audit=load(AUDIT)
    if audit['instructions_found']:raise ValueError('Read applicable project instructions before housekeeping')
    rows={r['path']:r for r in audit['source_rows']};selected=[];blocked=[]
    proposed=[(ROOT/'validation'/name,'completed_source_generator',GENERATORS[name]) for name in GENERATORS]
    proposed += [(ROOT/'validation'/name,'completed_fixed_case_source_diagnostic',[]) for name in INSPECTORS]
    proposed += [(ROOT/'main.py','unused_IDE_sample',[]),(ROOTS[0]/'coordinator/planner.py','unused_empty_extension_placeholder',[])]
    for p,kind,target_names in proposed:
        row=rows.get(str(p))
        if not row:raise ValueError('Audit does not contain exact selected source')
        active_refs=[ref for ref in row['referenced_by'] if Path(ref).suffix in SOURCE_SUFFIX and 'source_cleanup' not in ref and Path(ref).name!='audit_legacy_source_cleanup_20261005.py']
        if row['protected_reasons'] or active_refs:
            blocked.append({'path':str(p),'reasons':row['protected_reasons'],'active_code_references':active_refs});continue
        if file_sha256(p)!=row['sha256']:raise ValueError('Source changed after read-only audit')
        targets=[ROOT/'validation'/name for name in target_names]
        if any(not t.is_file() for t in targets):raise ValueError('A generated final implementation is missing')
        if kind=='unused_IDE_sample' and not('print_hi' in p.read_text(encoding='utf-8') and 'PyCharm' in p.read_text(encoding='utf-8')):raise ValueError('IDE sample no longer matches')
        if kind=='unused_empty_extension_placeholder' and any(not s.strip().startswith('#') for s in p.read_text(encoding='utf-8').splitlines() if s.strip()):raise ValueError('Planner extension has acquired live code')
        selected.append(record(p,kind,'No current import,code/config caller or frozen-source SHA;exact bytes archived before removal',targets))
    # Completed root debug output is archived only when no source/config consumes it.
    output_names=['_pwr.txt','_pwr2.txt','agent_eval_run.log','agent_eval_v2.log','retry_scenarios.log',
                  'demo_emilia.png','demo_gorkha.png','prediction_Emilia-Romagna_Italy_32.png',
                  'prediction_Emilia-Romagna_Italy_32_sample5.png','prediction_Gorkha_Nepal_32_sample5.png',
                  'detect_gorkha.json','pipeline_final.json','pipeline_gorkha.json','test_report.png']
    code_paths=[Path(r['path']) for r in audit['source_rows'] if Path(r['path']).name!='audit_legacy_source_cleanup_20261005.py']
    config_paths=[p for root in ROOTS for p in (root/'configs').rglob('*') if p.is_file() and p.suffix in ('.yaml','.yml','.json')]
    code_text={str(p):p.read_text(encoding='utf-8-sig',errors='replace') for p in code_paths+config_paths}
    authority_dirs=[root/name for root in ROOTS for name in ('datasets','experiments','models_prior','validation')]
    authority_hashes=set()
    for folder in authority_dirs:
        for p in folder.rglob('*.json'):
            if p.stat().st_size<=4*1024**2:authority_hashes.update(re.findall(r'(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])',p.read_text(encoding='utf-8-sig',errors='replace')))
    for name in output_names:
        p=ROOT/name
        if not p.exists():continue
        used=[path for path,text in code_text.items() if name in text]
        if used or file_sha256(p) in authority_hashes:
            blocked.append({'path':str(p),'active_code_references':used,'reason':'referenced output or frozen hash retained'});continue
        selected.append(record(p,'completed_root_debug_output','Historic generated debug output;exact bytes archived,all structured study results retained'))
    compiled=[]
    skip={'.git','.idea','.aws','.agents','.codex','cache','datasets','experiments','models_prior','artifact_runtime','node_modules'}
    for root in ROOTS:
        for base,dirs,names in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in skip and not Path(base,d).is_symlink()]
            if Path(base).name!='__pycache__':continue
            for name in names:
                p=Path(base,name)
                if p.suffix=='.pyc':compiled.append(record(p,'regenerable_project_bytecode','Project Python bytecode;original Python source retained'))
    # Never serialize user credentials or remove source datasets/weights/documents.
    if any(x['relative'].startswith(('datasets/','models_prior/','竞赛材料/')) or '.env' in Path(x['path']).name for x in selected+compiled):raise ValueError('Protected user artifacts entered housekeeping')
    archive=OUT/'retired_sources_and_debug_outputs.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for row in selected:z.write(row['path'],row['archive_member'])
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive CRC verification failed')
        for row in selected:
            if hashlib.sha256(z.read(row['archive_member'])).hexdigest()!=row['sha256']:raise ValueError('Archived source changed')
    before_runtime=load(ROOT/'validation/hardgap_closure_verification_20261005.json')['protected_runtime_artifact_sha256']
    protected_sources={str(p):file_sha256(p) for p in code_paths if str(p) not in {row['path'] for row in selected}}
    plan={'schema_version':1,'created_at':datetime.now().isoformat(timespec='seconds'),'roots':[str(r) for r in ROOTS],
          'user_authorization':'clear unused historical code and files;preserve active implementation and necessary scientific replay',
          'archive':{'path':str(archive),'sha256':file_sha256(archive),'bytes':archive.stat().st_size,'contents_sha256_verified':True},
          'archived_files_to_remove':selected,'regenerable_bytecode_to_remove':compiled,'candidates_retained_after_dependency_checks':blocked,
          'protected_remaining_source_sha256':protected_sources,'protected_runtime_artifact_sha256':before_runtime,
          'source_data_labels_protocols_study_results_and_current_weights_removed':False,
          'registry_sha256_before':file_sha256(DATA/'mine_case_registry.json'),'state_counts_before':{
              k:load(DATA/'STATE.json')[k] for k in ('completed_native_training_runs','completed_lightweight_feature_diagnostics')},
          'reserved_model_predictions_before':{c:registry()['cases'][c]['model_predictions_run'] for c in ('CL0002','KG0001','US_BC2021','TR_COP2024')}}
    write(OUT/'plan.json',plan)
    print(json.dumps({'archive_files':len(selected),'source_files':sum(r['kind']!='completed_root_debug_output' for r in selected),
        'debug_outputs':sum(r['kind']=='completed_root_debug_output' for r in selected),'bytecode_files':len(compiled),
        'archive_KiB':round(archive.stat().st_size/1024,1),'planned_original_bytes':sum(r['bytes'] for r in selected+compiled),
        'retained_despite_candidate_status':blocked},ensure_ascii=False,indent=2))
if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
