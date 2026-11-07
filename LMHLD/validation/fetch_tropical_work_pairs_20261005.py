"""Use the original bounded satellite reader with the frozen tropical plan."""
from pathlib import Path
import json
import sys

ROOT = Path('D:/LMHLD')
sys.path.insert(0, str(ROOT / 'validation'))
from fetch_diverse_operation_pairs import run

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    output = ROOT / 'datasets/tropical_work_pairs_20261005'
    parent = json.loads((output / 'site_plan.json').read_text(encoding='utf-8'))
    failures = {}
    for case, site in parent['sites'].items():
        try:
            run(out=output, plan={**parent, 'sites': {case: site}})
        except Exception as exc:
            failures[case] = {'error': type(exc).__name__, 'reason': str(exc),
                              'pixel_labels_admitted': False, 'model_predictions': 0}
            print(case, 'acquisition pending:', str(exc), flush=True)
    (output / 'acquisition_pending.json').write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
