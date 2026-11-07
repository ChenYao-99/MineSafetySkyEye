from pathlib import Path
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT/'validation'))
from fetch_diverse_operation_pairs import run
OUT=ROOT/'datasets/slats_operation_development_20261005'
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');run(out=OUT,plan=json.loads((OUT/'site_plan.json').read_text(encoding='utf-8')))
