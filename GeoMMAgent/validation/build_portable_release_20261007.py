"""Copy a bounded runtime and its default artifacts, without dataset junctions."""
from pathlib import Path
from datetime import datetime
import importlib.util
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
ML = Path('D:/LMHLD')
OUT = ROOT / 'cache/portable_task_acceptance_20261007'
RELEASE = OUT / '迁移验收 空格'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(destination=None):
    global OUT,RELEASE
    if destination is not None:
        OUT=Path(destination).resolve()
        assert OUT.is_relative_to((ROOT/'cache').resolve())
        OUT.mkdir(parents=True,exist_ok=True);RELEASE=OUT/'迁移验收 空格'
    sys.path.insert(0, str(ROOT / 'validation'))
    from build_isolated_release_20261006 import copy_runtime
    agent, detector = RELEASE / 'GeoMMAgent', RELEASE / 'LMHLD'
    copy_runtime(ROOT, agent, ['backend.py', 'coordinator', 'configs', 'exec_agents', 'toolkit', 'geomm_mcp',
                              'server_runtime.py', 'case_archive.py', 'runtime_identity.py'])
    copy_runtime(ML, detector, ['detector_service.py', 'knowledge_base.py', 'geo_tools.py',
                              'server_runtime.py', 'runtime_identity.py'])
    for name in ['knowledge_sources.json', 'knowledge_chunks.json', 'model_registry.json']:
        shutil.copy2(ML / name, detector / name)
    for name in ['矿安天眼_前端演示(后端版).html', 'kat_demo.html', 'start_demo.ps1', 'stop_demo.ps1', 'LICENSE', 'NOTICE']:
        shutil.copy2(ROOT / name, agent / name)
    (agent/'deployment').mkdir(exist_ok=True)
    for name in ['doctor.py','restore_archive.py']:
        shutil.copy2(ROOT/'deployment'/name,agent/'deployment'/name)
    shutil.copy2(ROOT/'archive_recovery.py',agent/'archive_recovery.py')
    if (ROOT / 'context_files').exists():
        shutil.copytree(ROOT / 'context_files', agent / 'context_files', dirs_exist_ok=True)
    (agent / 'resources').mkdir(exist_ok=True)
    token = Path('D:/Miniconda3/Lib/site-packages/dashscope/resources/qwen.tiktoken')
    shutil.copy2(token, agent / 'resources/qwen.tiktoken')
    weight = Path('real_ft_T1b_ours_seed42/best_model.pth')
    (detector / 'models_prior' / weight.parent).mkdir(parents=True, exist_ok=True)
    shutil.copy2(ML / 'models_prior' / weight, detector / 'models_prior' / weight)
    registry = Path('datasets/mine_benchmark_20261002/mine_case_registry.json')
    (detector / registry.parent).mkdir(parents=True, exist_ok=True)
    shutil.copy2(ML / registry, detector / registry)
    auxiliary=Path('datasets/input_reserve_registry.json')
    if (ML/auxiliary).exists():shutil.copy2(ML/auxiliary,detector/auxiliary)
    work_index = Path('datasets/slats_operation_development_20261005/runtime_work_context_registry.json')
    work = json.loads((ML / work_index).read_text(encoding='utf-8'))
    for entry in work['entries'].values():
        for field in ['reference', 'mask']:
            source = Path(entry[field])
            relative = source.relative_to(ML)
            target = detector / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            assert sha(source) == entry[field+'_sha256']
            entry[field] = str(target)
    (detector / work_index.parent).mkdir(parents=True, exist_ok=True)
    (detector / work_index).write_text(json.dumps(work, ensure_ascii=False, indent=2), encoding='utf-8')
    files = {}
    for p in RELEASE.rglob('*'):
        relative=p.relative_to(RELEASE)
        if not p.is_file() or p.suffix=='.pyc' or '__pycache__' in p.parts:continue
        if len(relative.parts)>1 and relative.parts[1] in ('cache','data','uploads','temp_images','results','run'):continue
        files[str(relative)]=sha(p)
    assert not any(name.endswith('.env') or name.startswith('GeoMMAgent/data/') for name in files)
    value = {'created_at': datetime.now().astimezone().isoformat(), 'release': str(RELEASE),
        'files': files, 'default_weight_SHA256': sha(detector / 'models_prior' / weight),
        'protected_case_registry_copied': True, 'work_reference_paths_relocated_and_hashes_preserved': True,
        'dataset_junctions_or_original_environment_packages_copied': False,
        'credentials_copied': False, 'scope': 'same machine, copied runtime and default assets;upload/default-model acceptance;not all optional models or another-machine qualification'}
    OUT.mkdir(exist_ok=True)
    (OUT / 'portable_release_manifest.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PORTABLE_RELEASE', str(RELEASE), len(files), 'files')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    build()
