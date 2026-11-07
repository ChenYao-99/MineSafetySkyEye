"""Preregistered paired-arm contrasts, separate from model/method selection."""
from pathlib import Path
import sys
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import rgb_native_training as exp


def assess(pairs, limits):
    keys = exp.METRICS
    before = {k: float(np.mean([a[k] for a, b in pairs])) for k in keys}
    after = {k: float(np.mean([b[k] for a, b in pairs])) for k in keys}
    delta = {k: after[k]-before[k] for k in keys}
    checks = {'small_coverage_gain': delta['small_cover50'] >= limits['min_cover_gain'],
              'strict_iou_gain': delta['small_iou50'] >= limits['min_iou_gain'],
              'f1_preserved': delta['f1'] >= -limits['max_f1_loss'],
              'macro_fpr_increase_control': delta['background_fpr'] <= limits['max_macro_fpr_increase'],
              'region_fpr_increase_control': max(b['background_fpr']-a['background_fpr'] for a, b in pairs) <= limits['max_region_fpr_increase'],
              'macro_absolute_fpr': after['background_fpr'] <= limits['max_macro_fpr'],
              'region_absolute_fpr': max(b['background_fpr'] for a, b in pairs) <= limits['max_region_fpr'],
              'candidate_count_control': after['false_components_per_mpix'] <= limits['max_candidate_ratio']*before['false_components_per_mpix']}
    return {'before': before, 'after': after, 'delta': delta, 'checks': checks, 'passed': all(checks.values())}


def register(out):
    out = Path(out); path = out/'primary_comparison_protocol.json'
    if path.exists(): return exp.base.read(path)
    if list(out.glob('*/fold_*/seed*/result.json')):
        raise ValueError('Pairwise contrast thresholds must be fixed before target inference')
    p = {'registered_at': exp.base.now(), 'all_target_results_absent_at_registration': True,
         'model_protocol_sha256': exp.base.sha(out/'protocol.json'),
         'analyzer_sha256': exp.base.sha(__file__),
         'tests_sha256': exp.base.sha(exp.ROOT/'tests/test_rgb_native_analysis.py'),
         'primary': ['fixed', 'tta5', 'scale_jitter', 'area_guard'],
         'descriptive_contrasts': [['fixed', 'tta5', 'scale_jitter', 'tta5'],
                                   ['fixed', 'multiscale_raw', 'scale_jitter', 'multiscale_raw'],
                                   ['fixed', 'tta5', 'fixed', 'area_guard'],
                                   ['scale_jitter', 'tta5', 'scale_jitter', 'area_guard']],
         'limits': {'min_cover_gain': .02, 'min_iou_gain': .02, 'max_f1_loss': .005,
                    'max_macro_fpr_increase': .002, 'max_region_fpr_increase': .01,
                    'max_macro_fpr': .03, 'max_region_fpr': .05, 'max_candidate_ratio': 1.10},
         'threshold_origin': 'Existing prior scale study and component acceptance guardrails, not newly selected from target results.',
         'interpretation': 'Require both seeds to pass. Diagnostic development evidence, not independent blind or field acceptance. Does not select or change any model.'}
    exp.base.write(path, p); return p


def run(out):
    out = Path(out); p = exp.base.read(out/'primary_comparison_protocol.json'); model = exp.base.read(out/'protocol.json')
    for path, digest in [(Path(__file__), p['analyzer_sha256']), (exp.ROOT/'tests/test_rgb_native_analysis.py', p['tests_sha256']), (out/'protocol.json', p['model_protocol_sha256'])]:
        if exp.base.sha(path) != digest: raise ValueError('Frozen comparison changed')
    contrasts = []
    for contrast in [p['primary'], *p['descriptive_contrasts']]:
        a, am, b, bm = contrast; rows = []
        for seed in model['seeds']:
            pairs = []
            for g in model['held_indices']:
                old = exp.base.read(exp.folder(out, a, g, seed)/'result.json')['methods'][am]['held_out']
                new = exp.base.read(exp.folder(out, b, g, seed)/'result.json')['methods'][bm]['held_out']
                pairs.append((old, new))
            rows.append({'seed': seed, **assess(pairs, p['limits'])})
        contrasts.append({'contrast': contrast, 'by_seed': rows, 'passed_both_seeds': all(r['passed'] for r in rows)})
    value = {'completed_at': exp.base.now(), 'protocol_sha256': exp.base.sha(out/'primary_comparison_protocol.json'),
             'primary': contrasts[0], 'descriptive_contrasts': contrasts[1:], 'production_changed': False}
    exp.base.write(out/'paired_comparisons.json', value); return value


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); parser.add_argument('--register', action='store_true')
    a = parser.parse_args(); (register if a.register else run)(Path(a.output))
