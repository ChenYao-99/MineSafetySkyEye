"""Read-only post-cleanup preservation verification; no generation/model requests."""
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def main():
    plan=HERE/'delete_manifest_20261016.json'
    compressed=plan.with_suffix('.json.gz')
    if plan.exists():manifest=json.loads(plan.read_text(encoding='utf-8'))
    else:
        with gzip.open(compressed,'rt',encoding='utf-8') as source:manifest=json.load(source)
    expected_doc_updates=HERE/'documentation_updates_20261016.json'
    documentation=json.loads(expected_doc_updates.read_text(encoding='utf-8-sig')) if expected_doc_updates.exists() else {}
    checks=hashes=metadata_only=0
    for name,record in manifest['protected'].items():
        path=Path(name)
        if not path.is_file():raise ValueError('Protected file removed: '+name)
        if name in documentation:
            if sha(path)!=documentation[name]['after_sha256']:raise ValueError('Cleanup documentation changed unexpectedly')
        elif 'sha256' in record:
            if sha(path)!=record['sha256']:raise ValueError('Protected hash changed: '+name)
            hashes+=1
        else:
            meta=path.stat()
            if meta.st_size!=record['bytes'] or meta.st_mtime_ns!=record['mtime_ns']:
                raise ValueError('Excluded original data changed: '+name)
            metadata_only+=1
        checks+=1
    for row in manifest['links_to_unlink_without_recursion']:
        if Path(row['path']).exists():raise ValueError('Retired junction remains')
        if not Path(row['target']).is_dir():raise ValueError('Shared junction target was deleted')
    for name,record in documentation.items():
        if sha(name)!=record['after_sha256']:raise ValueError('Document link update no longer matches receipt')
    remaining=[row['path'] for row in manifest['files'] if Path(row['path']).exists()]
    if remaining:raise ValueError('Deletion target remains: '+remaining[0])
    result={'passed':True,'protected_file_checks':checks,'unchanged_hash_checks':hashes,
            'original_data_metadata_checks':metadata_only,'intentional_documentation_updates':list(documentation),
            'shared_link_targets_preserved':True,'deletion_manifest_fully_applied':True,
            'default_models_and_production_case_archive_preserved':True,
            'new_predictions_or_cloud_calls':0}
    (HERE/'preservation_verification_20261016.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
