"""Require corrected source regressions for families affected by the colour protocol bug."""
from pathlib import Path
import json
from model_registry import file_sha256
ROOT=Path(__file__).resolve().parents[1]
AFFECTED_FAMILIES={'rgb_context_residual_pilot_20261004','rgb_colour_style_pilot_20261004',
                  'rgb_soft_prior_pilot_20261004','rgb_direct_context_20261004','winter_normal_study_20261004'}


def load_family_rows(folder):
    folder=Path(folder);original=json.loads((folder/'results.json').read_text(encoding='utf-8'))
    sidecar=folder/'source_evaluation_correction.json'
    if not sidecar.exists():
        if folder.name in AFFECTED_FAMILIES:raise ValueError('Corrected source regression required for affected family')
        return original['models']
    status=json.loads(sidecar.read_text(encoding='utf-8'))
    if status.get('source_RGB_original_evaluation_valid') is not False:raise ValueError('Unknown source correction status')
    corrected_path=Path(status['replacement_source_regression'])
    corrected=json.loads(corrected_path.read_text(encoding='utf-8'))
    digest=file_sha256(folder/'results.json')
    replacements={r['id']:r for r in corrected['models'] if r['family']==folder.name}
    rows=[]
    for old in original['models']:
        if old['id'] not in replacements:raise ValueError('Incomplete corrected source regression')
        row=replacements[old['id']]
        if row['historical_record_sha256']!=digest or row['weights_sha']!=old['weights_sha']:
            raise ValueError('Correction references changed result/weights')
        if row['old_reference']!=old['old_reference'] or row['reviewed_normal_activation']!=old['reviewed_normal_activation']:
            raise ValueError('Source correction unexpectedly changed native metrics')
        if row['corrected_source_protocol_sha256']!=corrected['protocol_sha256']:
            raise ValueError('Correction protocol inconsistent')
        rows.append(row)
    return rows
