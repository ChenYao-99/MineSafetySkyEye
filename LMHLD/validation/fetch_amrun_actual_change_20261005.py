"""Fetch the pre-registered actual-work pair with the shared original-DN downloader."""
from pathlib import Path
import argparse, json, sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT/'validation'))
from fetch_diverse_operation_pairs import run
OUT=ROOT/'datasets/source_backed_operation_change_20261005'

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--full-window',action='store_true');args=parser.parse_args()
    if args.full_window:OUT=OUT/'full_published_geometry_window_v2'
    run(out=OUT,plan=json.loads((OUT/'site_plan.json').read_text(encoding='utf-8')))
