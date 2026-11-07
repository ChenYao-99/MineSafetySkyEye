"""Original-input visual QA only; contrast stretch is display-only and never training data."""
from pathlib import Path
import json,sys
import numpy as np
import rasterio
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from model_registry import file_sha256
from prepare_winter_constructed_normals import display
OUT=ROOT/'datasets/diverse_operation_pairs_20261004'
plan=json.loads((OUT/'site_plan.json').read_text(encoding='utf-8'))
summary={}
for site in plan['sites']:
    folder=OUT/site;record=json.loads((folder/'acquisition.json').read_text(encoding='utf-8'));row={}
    for period in ('pre','post'):
        image,valid=display(folder/f'{period}_BGRN_10m.tif');original=folder/(period+'_original_preview.png')
        Image.fromarray(image).save(original)
        with rasterio.open(folder/f'{period}_SCL_10m.tif') as f:scl=f.read(1)
        if file_sha256(folder/f'{period}_BGRN_10m.tif')!=record['images'][period]['sha256']:raise ValueError('Original input changed')
        row[period]={'date':record['images'][period]['datetime'],'shape':list(image.shape[:2]),'valid_fraction':float(valid.mean()),
                     'SCL_weather_obscured_fraction':float(np.isin(scl,[0,1,3,8,9,10,11]).mean()),
                     'preview':str(original),'preview_sha256':file_sha256(original)}
    summary[site]=row
(OUT/'original_preview_manifest.json').write_bytes((json.dumps(summary,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
sys.stdout.reconfigure(encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))
