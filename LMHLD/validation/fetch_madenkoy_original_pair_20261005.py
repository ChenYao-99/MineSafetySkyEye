from pathlib import Path
import sys,json
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT/'validation'))
from fetch_diverse_operation_pairs import run
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');out=ROOT/'datasets/madenkoy_development_20261005'
    run(out=out,plan=json.loads((out/'site_plan.json').read_text(encoding='utf-8')))
