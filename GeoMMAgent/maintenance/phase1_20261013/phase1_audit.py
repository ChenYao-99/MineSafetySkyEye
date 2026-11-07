"""Freeze a small baseline and qualify existing data without model inference.

Run with D:\\Miniconda3\\envs\\lmhld\\python.exe. The immutable freeze is
created once. Subsequent runs verify it instead of overwriting its evidence.
No service, original reference, model, or reserved image is modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

GEO = Path(r'D:\GeoMMAgent')
DET = Path(r'D:\LMHLD')
HERE = GEO / 'maintenance/phase1_20261013'
DATA = DET / 'datasets/phase1_20261013'


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_record(path):
    p = Path(path)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def same_instant(left, right):
    try:
        a = datetime.fromisoformat(left.replace('Z', '+00:00'))
        b = datetime.fromisoformat(right.replace('Z', '+00:00'))
        return a.tzinfo is not None and b.tzinfo is not None and a == b
    except (ValueError, TypeError, AttributeError):
        return False


def bounds_intersect(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def reserve_overlap(bounds):
    registry = read(DET / 'datasets/input_reserve_registry.json')
    protected = [(key, value['protected_geographic_extent_wgs84'])
                 for key, value in registry['cases'].items()]
    geometry = read(DET / 'validation/reserved_geographic_protection_20261005.json')
    protected += [(x['case_id'], x['bounds_wgs84']) for x in geometry['cases']]
    return sorted({case_id for case_id, extent in protected if bounds_intersect(bounds, extent)})


def snapshot():
    """Freeze current source; keep large model and submission files by hash."""
    manifest_path = HERE / 'baseline.json'
    if manifest_path.exists():
        frozen = read(manifest_path)
        changed = [x['path'] for x in frozen['source_files'] + frozen['model_files']
                   if not Path(x['path']).exists() or sha(x['path']) != x['sha256']]
        archive = frozen['source_archive']
        archive_ok = Path(archive['path']).exists() and sha(archive['path']) == archive['sha256']
        custody = frozen['reserve_registry']
        custody_ok = sha(custody['path']) == custody['sha256']
        return dict(existing_freeze=True, changed=changed,
                    freeze=str(manifest_path), model_files=len(frozen['model_files']),
                    source_archive_intact=archive_ok, reserve_registry_unchanged=custody_ok)
    dependency_list = read(GEO / 'maintenance/slim_project/runtime_dependencies.json')
    paths = set()
    for root_name, data in dependency_list.items():
        root = Path(root_name)
        for rel in data['runtime_python_files']:
            p = root / rel
            if p.exists():
                paths.add(p)
    for root in [GEO, DET]:
        for name in ['README.md', 'LICENSE', 'NOTICE', 'requirements.txt',
                     'runtime_identity.py', 'model_registry.json']:
            if (root / name).is_file():
                paths.add(root / name)
    for name in ['矿安天眼_前端演示(后端版).html', 'kat_demo.html', 'task_study.html',
                 'start_demo.ps1', 'stop_demo.ps1', '启动矿安天眼演示.bat']:
        paths.add(GEO / name)
    for p in (GEO / 'configs').glob('*.yaml'):
        paths.add(p)
    for p in (GEO / 'deployment').iterdir():
        if p.is_file():
            paths.add(p)
    for p in (DET / 'models_prior').glob('*/adapter.json'):
        paths.add(p)
    # Knowledge and detector configuration without sample arrays or databases.
    for p in [DET / 'knowledge_base.py', DET / 'knowledge_sources.json', DET / 'knowledge_chunks.json']:
        if p.exists():
            paths.add(p)
    source_files = [file_record(p) for p in sorted(paths)]
    weights = sorted((DET / 'models_prior').rglob('*.pth'))
    weights.append(DET / 'best_model_Emilia-Romagna_Italy_32.pth')
    model_files = [file_record(p) for p in weights if p.exists()]
    registry = read(DET / 'model_registry.json')
    registry_checks = []
    for name, item in registry['artifacts'].items():
        p = DET / 'models_prior' / item['path']
        registry_checks.append(dict(name=name, path=str(p), passed=p.exists() and sha(p) == item['sha256']))
    if not all(x['passed'] for x in registry_checks):
        raise RuntimeError('Model registry identity mismatch; refusing to freeze')
    custody = DET / 'datasets/input_reserve_registry.json'
    baseline_archive = HERE / 'baseline_source.zip'
    with ZipFile(baseline_archive, 'x', compression=ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(paths):
            root = GEO if p.is_relative_to(GEO) else DET
            z.write(p, root.name + '/' + p.relative_to(root).as_posix())
    submission_dir = Path.home() / 'Desktop' / '2026AIC'
    submitted_candidates = [p for p in submission_dir.iterdir()
                            if p.is_file() and ('创新方案.pdf' in p.name or '答辩PPT.pdf' in p.name or '演示视频.mp4' in p.name)]
    prior_archive = GEO / '竞赛材料/output/source/矿安天眼_源码与实验索引_审阅包.zip'
    comparison = dict(path=str(prior_archive), basis='earlier review archive; not verified uploaded code archive')
    if prior_archive.exists():
        with ZipFile(prior_archive) as z:
            common = []
            for x in source_files:
                p = Path(x['path']); root = GEO if p.is_relative_to(GEO) else DET
                key = root.name + '/' + p.relative_to(root).as_posix()
                if key in z.namelist():
                    common.append(dict(archive_member=key, same=hashlib.sha256(z.read(key)).hexdigest() == x['sha256']))
            comparison.update(common_files=len(common), unchanged=sum(x['same'] for x in common),
                              differences=[x['archive_member'] for x in common if not x['same']])
    frozen = dict(schema_version=1, frozen_at=now(),
                  scope='2026-10-13 current workspace baseline; user-reported submitted material candidates hashed separately',
                  uploaded_source_archive_identity_verified=False,
                  source_files=source_files, source_archive=file_record(baseline_archive),
                  model_files=model_files, registry_checks=registry_checks,
                  default_artifact=registry['artifacts']['upload:mine'],
                  submitted_material_candidates=[file_record(p) for p in sorted(submitted_candidates)],
                  prior_review_archive_comparison=comparison,
                  reserve_registry=file_record(custody),
                  excluded=['.env and credentials', 'case database', 'weights duplication', 'original image duplication'])
    write(manifest_path, frozen)
    return dict(existing_freeze=False, freeze=str(manifest_path), source_files=len(source_files),
                model_files=len(model_files), archive_bytes=baseline_archive.stat().st_size,
                prior_comparison=comparison)


def requested_use(case, purpose):
    """Fail closed for known role confusion before any future model work."""
    if case.get('reserved') or case.get('role') in ['reserved_final_test', 'development_normal_holdout']:
        return False, 'reserved_or_held_out_case'
    if not case.get('integrity_passed'):
        return False, 'source_or_asset_integrity_failed'
    if not case.get('paired_input_passed'):
        return False, 'input_quality_failed'
    kind = case['reference_kind']
    if purpose == 'accident_positive_core':
        return (kind == 'published_partial_accident_core' and case.get('positive_pixels', 0) > 0
                and case.get('chronology_passed') and case.get('existing_scope_admitted')), 'positive_core_only'
    if purpose == 'work_context':
        return (kind == 'government_mining_clearing_weak_context' and case.get('context_pixels', 0) > 0
                and case.get('existing_scope_admitted')), 'not_accident_negative_truth'
    if purpose == 'accident_negative':
        return (kind == 'independently_verified_non_failure_reference'
                and bool(case.get('verified_accident_negative'))), 'requires_independent_negative_evidence'
    if purpose == 'whole_accident_evaluation':
        return (kind == 'independent_scoped_complete_accident_reference'
                and bool(case.get('full_boundary_verified'))), 'requires_complete_scoped_accident_reference'
    return False, 'unsupported_purpose'


def pair_audit(case_dir):
    import numpy as np
    import rasterio
    case_dir = Path(case_dir)
    acquisition = read(case_dir / 'acquisition.json')
    observations = {}
    opened = {}
    files = []
    all_hashes_match = True
    for epoch in ['pre', 'post']:
        row = acquisition['images'][epoch]
        expected = row.get('assets_sha256') or {
            f'{epoch}_BGRN_10m.tif': row['sha256'],
            f'{epoch}_SCL_10m.tif': row['scl_sha256'],
            f'{epoch}_product.xml': row['product_XML_sha256'],
        }
        for name, digest in expected.items():
            p = case_dir / name
            ok = p.exists() and sha(p) == digest
            files.append(dict(path=str(p), sha256=digest, passed=ok))
            all_hashes_match &= ok
        with rasterio.open(case_dir / f'{epoch}_BGRN_10m.tif') as ds:
            a = ds.read()
            opened[epoch] = dict(shape=ds.shape, crs=ds.crs, transform=ds.transform)
            from rasterio.warp import transform_bounds
            extent = list(transform_bounds(ds.crs, 'EPSG:4326', *ds.bounds))
            protected_overlap = reserve_overlap(extent)
            usable = np.all(a != 0, axis=0)
            with rasterio.open(case_dir / f'{epoch}_SCL_10m.tif') as scl_ds:
                scl = scl_ds.read(1)
                scl_same_grid = ds.shape == scl_ds.shape and ds.crs == scl_ds.crs and ds.transform == scl_ds.transform
                usable &= np.isin(scl, [4, 5, 6, 7])
            xml_file = case_dir / f'{epoch}_product.xml'
            from xml.etree import ElementTree as ET
            xml = ET.parse(xml_file)
            xml_dates = [e.text for e in xml.iter() if e.tag.split('}')[-1] == 'PRODUCT_START_TIME']
            timestamp = row['datetime']
            tag_time = ds.tags().get('acquisition_datetime', '')
            date_verified = bool(xml_dates) and same_instant(xml_dates[0], timestamp) and same_instant(tag_time, timestamp)
            calibration = row['calibration']
            scale_match = np.allclose(ds.scales, calibration['scales'])
            offsets_match = np.allclose(ds.offsets, calibration['reflectance_offsets'])
            observations[epoch] = dict(datetime=timestamp, xml_datetime=xml_dates,
                time_verified=date_verified, bands=list(ds.descriptions), crs=str(ds.crs),
                bounds_wgs84=extent, protected_overlaps=protected_overlap,
                shape=list(ds.shape), res=list(ds.res),
                raw_dn_preserved=(ds.tags().get('raw_DN_values') == 'preserved' or ds.tags().get('raw_DN_samples') == 'preserved'),
                calibration_matches=bool(scale_match and offsets_match), scl_same_grid=scl_same_grid,
                observable_fraction=float(usable.mean()))
            opened[epoch]['observable'] = usable
    same_grid = all(opened['pre'][k] == opened['post'][k] for k in ['shape', 'crs', 'transform'])
    common = opened['pre']['observable'] & opened['post']['observable'] if same_grid else None
    metadata_ok = all(x['time_verified'] and x['calibration_matches'] and x['scl_same_grid']
                      and x['raw_dn_preserved'] and x['bands'] == ['B02', 'B03', 'B04', 'B08']
                      and not x['protected_overlaps']
                      for x in observations.values())
    return dict(files=files, integrity_passed=bool(all_hashes_match), epochs=observations,
                same_grid=same_grid, paired_input_passed=bool(same_grid and metadata_ok),
                common_observable_fraction=float(common.mean()) if common is not None else None), common, opened['post']


def qualify_existing():
    import numpy as np
    import rasterio
    cases = []
    problems = []
    gams_dir = DET / 'datasets/arid_accident_operation_20261004/ZA_GAMS2020S'
    ref_dir = gams_dir / 'pixel_reference_v1_wang2026'
    ref = read(ref_dir / 'reference.json')
    audit, common, grid = pair_audit(gams_dir)
    dependencies = []
    for path, digest in ref['frozen_dependencies'].items():
        p = Path(path)
        dependencies.append(dict(path=str(p), expected_sha256=digest, passed=p.exists() and sha(p) == digest))
    for name, digest in ref['frozen_assets'].items():
        p = ref_dir / name
        dependencies.append(dict(path=str(p), expected_sha256=digest, passed=p.exists() and sha(p) == digest))
    with rasterio.open(ref_dir / 'labels_partial.tif') as ds:
        labels = ds.read(1)
        label_grid_ok = ds.shape == grid['shape'] and ds.crs == grid['crs'] and ds.transform == grid['transform']
    values, counts = np.unique(labels, return_counts=True)
    label_counts = {str(int(k)): int(v) for k, v in zip(values, counts)}
    event = ref['event_date']
    pre = audit['epochs']['pre']['datetime'][:10]
    post = audit['epochs']['post']['datetime'][:10]
    gams = dict(case_id='ZA_GAMS2020S', mine_group='ZA_GAMSBERG', role='development', reserved=False,
        reference_kind='published_partial_accident_core', reference_path=str(ref_dir / 'reference.json'),
        reference_sha256=sha(ref_dir / 'reference.json'), existing_scope_admitted=ref['accepted_for_development_positive_core_supervision'],
        reference_new_in_this_phase=False, full_boundary_verified=False, verified_accident_negative=False,
        event_date=event, chronology_passed=pre < event < post,
        positive_pixels=label_counts.get('1', 0), background_pixels=label_counts.get('0', 0),
        unknown_pixels=label_counts.get('255', 0), label_counts=label_counts,
        label_grid_matches=label_grid_ok, core_observable_pixels=int(np.count_nonzero((labels == 1) & common)),
        integrity_passed=audit['integrity_passed'] and all(x['passed'] for x in dependencies),
        paired_input_passed=audit['paired_input_passed'] and label_grid_ok,
        pair_audit=audit, source_checks=dependencies, independent_professional_review=False,
        usage_note='post-event positive core only; no pre-event negative labels; outside core unknown; not new full boundary',
        labels_path=str(ref_dir / 'labels_partial.tif'), confidence_path=str(ref_dir / 'confidence_partial.tif'),
        unresolved=['original source map base-image date unknown', 'no original survey GIS', 'full failure/runout extent unverified',
                    'public redistribution permissions of digitised paper outline need separate review'])
    gams['allowed_uses'] = {p: bool(requested_use(gams, p)[0]) for p in ['accident_positive_core', 'work_context', 'accident_negative', 'whole_accident_evaluation']}
    cases.append(gams)
    pending = read(ref_dir / 'reference_review_pending.json')
    problems.append(dict(case_id='ZA_GAMS2020S', type='superseded_pending_snapshot',
        old=str(ref_dir / 'reference_review_pending.json'), authoritative=str(ref_dir / 'reference.json'),
        pending_claim=pending['accepted_for_development_positive_core_supervision'],
        current_claim=ref['accepted_for_development_positive_core_supervision'],
        resolution='use formal scoped reference plus hash verification; historical files kept unchanged'))
    work_specs = [
        ('QLD_ARID_46790_46791', DET / 'datasets/data_conditions_20261007/SLATS_arid_pairs/QLD_ARID_46790_46791', 'source_work_reference'),
        ('QLD_ARID_45682', DET / 'datasets/data_conditions_20261007/SLATS_arid_pairs/QLD_ARID_45682', 'source_work_reference'),
        ('AU_ANDOOM_LOCAL_SLATS2324', DET / 'datasets/tropical_work_pairs_local_quality_20261005/AU_ANDOOM_LOCAL_SLATS2324', 'work_reference_v1_SLATS'),
    ]
    acceptance = read(DET / 'datasets/tropical_work_pairs_20261005/work_reference_acceptance.json')
    for case_id, case_dir, sub in work_specs:
        rd = case_dir / sub
        r = read(rd / 'reference.json')
        a, common, grid = pair_audit(case_dir)
        mask_path = rd / 'expected_work_mask.tif'
        expected = r.get('mask_sha256') or r['frozen_assets']['expected_work_mask.tif']
        with rasterio.open(mask_path) as ds:
            mask = ds.read(1)
            mask_grid = ds.shape == grid['shape'] and ds.crs == grid['crs'] and ds.transform == grid['transform']
        default_source = case_dir.parent / (case_id + '_official_mining_polygons.geojson')
        src_path = Path(r.get('source_original_geojson', default_source))
        source_matches = src_path.exists() and sha(src_path) == r['source_sha256']
        pixels = int(np.count_nonzero(mask == 1))
        observable = int(np.count_nonzero((mask == 1) & common))
        existing_admission = r['status'] == 'qualified_source_backed_expected_work'
        if case_id in acceptance['accepted']:
            admitted = acceptance['accepted'][case_id]['references'][0]
            existing_admission = admitted['sha256'] == sha(rd / 'reference.json')
            problems.append(dict(case_id=case_id, type='draft_reference_with_signed_hash_acceptance',
                authoritative=str(DET / 'datasets/tropical_work_pairs_20261005/work_reference_acceptance.json'),
                resolution='aggregate acceptance binds the exact reference hash; keep raw draft record unchanged'))
        c = dict(case_id=case_id, mine_group=r['mine_group'], role='auxiliary_development', reserved=False,
            reference_kind='government_mining_clearing_weak_context', reference_path=str(rd / 'reference.json'),
            reference_sha256=sha(rd / 'reference.json'), reference_new_in_this_phase=False,
            existing_scope_admitted=existing_admission, full_boundary_verified=False,
            verified_accident_negative=False, chronology_passed=a['epochs']['pre']['datetime'] < a['epochs']['post']['datetime'],
            context_pixels=pixels, core_observable_pixels=observable,
            paired_input_passed=a['paired_input_passed'] and mask_grid and pixels == observable,
            integrity_passed=a['integrity_passed'] and sha(mask_path) == expected and source_matches and pixels == r['expected_work_pixels'],
            pair_audit=a, source_geojson=str(src_path), source_hash_passed=source_matches,
            source_nominal_period=r['source_nominal_period'], mask_path=str(mask_path),
            confidence=r.get('confidence', 0.25), confidence_is_calibrated_probability=False,
            independent_professional_review=False,
            usage_note='mining-clearing context only; not confirmed absence of accident or stability; outside selected core unknown',
            unresolved=['exact clearing dates not known', 'no contemporaneous site safety/incident records',
                        'weak source-context label must not be relabelled as accident-negative'])
        c['allowed_uses'] = {p: bool(requested_use(c, p)[0]) for p in ['accident_positive_core', 'work_context', 'accident_negative', 'whole_accident_evaluation']}
        cases.append(c)
    for case_id, folder, reg in [
        ('TR_MAD2016', 'madenkoy_development_20261005', 'field_registration_v2_scale_matched'),
        ('DZ_KEF2022', 'kef_essnoun_development_20261005', 'reference_registration_v4_mutual_sift'),
    ]:
        p = DET / 'datasets' / folder / case_id / reg / 'registration.json'
        d = read(p)
        passed = d.get('both_dates_registration_pass', d.get('both_original_dates_gate_pass', False))
        cases.append(dict(case_id=case_id, role='development', reserved=False,
            reference_kind='rejected_published_outline_registration', reference_new_in_this_phase=False,
            paired_input_passed=False, integrity_passed=True, registration_report=file_record(p),
            existing_registration_passed=passed,
            allowed_uses={x: False for x in ['accident_positive_core', 'work_context', 'accident_negative', 'whole_accident_evaluation']},
            decision='no new registration search; reopen only with original georeferenced source or independently anchored new data',
            unresolved=['source-to-original-image registration failed', 'cannot use plotted outlines as native pixel truth']))
    report = dict(schema_version=1, reviewed_at=now(),
        scope='existing development data qualification; zero classifier predictions, fits or holdout opens',
        cases=cases, status_resolution=problems,
        reserved_registry=file_record(DET / 'datasets/input_reserve_registry.json'),
        reserve_note='registry is authoritative; no reserved image or boundary was opened by this audit',
        totals=dict(existing_partial_accident_cases=sum(x['allowed_uses']['accident_positive_core'] for x in cases),
                    existing_partial_positive_pixels=gams['positive_pixels'],
                    existing_weak_work_cases=sum(x['allowed_uses']['work_context'] for x in cases),
                    existing_weak_context_pixels=sum(x.get('context_pixels', 0) for x in cases if x['allowed_uses']['work_context']),
                    new_complete_accident_boundaries=0, new_verified_accident_negative_cases=0,
                    new_model_predictions=0, new_fits=0))
    write(DATA / 'catalog.json', report)
    usable = [dict(case_id=x['case_id'], role=x['role'], allowed_uses=x['allowed_uses'],
                   reference_path=x['reference_path'], reference_sha256=x['reference_sha256'],
                   pair_files=[z for z in x['pair_audit']['files'] if z['path'].endswith('_BGRN_10m.tif')],
                   labels_path=x.get('labels_path'), confidence_path=x.get('confidence_path'),
                   work_context_mask=x.get('mask_path'),
                   labels_apply_to='post_only' if x['allowed_uses']['accident_positive_core'] else 'weak_context_only',
                   unknown_must_be_excluded=True)
              for x in cases if any(x['allowed_uses'].values())]
    write(DATA / 'eligible_inputs.json', dict(created_at=now(), inputs=usable,
          not_a_new_training_authorisation=True, full_accident_or_safe_negative_evaluation_available=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['freeze', 'qualify', 'verify'])
    args = parser.parse_args()
    if args.operation in ['freeze', 'verify']:
        result = snapshot()
    else:
        report = qualify_existing()
        result = dict(catalog=str(DATA / 'catalog.json'), totals=report['totals'],
                      case_status=[dict(case=x['case_id'], integrity=x['integrity_passed'],
                                        input=x['paired_input_passed'], allowed=x['allowed_uses']) for x in report['cases']])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.operation in ['freeze', 'verify'] and result.get('existing_freeze'):
        if result['changed'] or not result['source_archive_intact'] or not result['reserve_registry_unchanged']:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
