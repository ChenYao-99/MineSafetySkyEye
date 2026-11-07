"""Rehydrate verified development-image displays; do not run any model."""
from pathlib import Path
import base64,copy,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import cv2,numpy as np,rasterio
OUT=Path('D:/GeoMMAgent/cache/high_value_sprint_20261008');OUT.mkdir(exist_ok=True)
sys.path.insert(0,'D:/GeoMMAgent')
from evidence_reporting import objective_facts


def main():
    source=ROOT/'experiments/same_input_reporting_workflow_20261005/detector_bank.json'
    source_digest=hashlib.sha256(source.read_bytes()).hexdigest()
    tasks=[]
    for task in json.loads(source.read_text(encoding='utf-8'))['tasks'][:6]:
        detection=copy.deepcopy(task['detection']);meta=detection['band_meta']
        paired='t1_ratio' in detection['stats'];epoch=meta['t2'] if paired else meta
        name=task['id'].split('_single')[0].split('_pair')[0];folder='Xinjing' if name=='CN_XJ2023' else name
        path=ROOT/'datasets/mine_benchmark_20261002/development'/folder/'post_BGRN_10m.tif'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==epoch['input_sha256']
        with rasterio.open(path) as image:
            pixels=image.read([1,2,3]).transpose(1,2,0).astype(np.float32);valid=np.all(image.read_masks([1,2,3])>0,axis=0)
            shape=[image.height,image.width]
        for channel in range(3):
            lo,hi=np.percentile(pixels[:,:,channel][valid],[2,98])
            pixels[:,:,channel]=np.clip((pixels[:,:,channel]-lo)/max(hi-lo,1e-6),0,1)*255
        pixels[~valid]=0
        ok,png=cv2.imencode('.png',pixels.astype(np.uint8));assert ok
        preview=base64.b64encode(png).decode()
        detection['size']=shape;detection['t2_rgb_b64' if paired else 'rgb_b64']=preview
        facts=objective_facts(detection)
        weight=detection['model_info']['artifact']['weights_sha256']
        dates=[value[:10] if value else 'unknown' for value in facts['capture_times']]
        expected={'capture_dates':', '.join(dates),'measurement':facts['change']['net_percentage_points'] if paired else facts['area_percent'],
            'unit':'percentage_points' if paired else 'percent','weight_sha256':weight,'accident_confirmed':'no'}
        tasks.append({'id':task['id'],'name':name,'region':'compare' if paired else 'upload','detection':detection,
            'expected':expected,'source_bank_sha256':source_digest,'raw_image_sha256':epoch['input_sha256'],
            'image_is_verified_same_raw_input_display':True,'new_model_predictions':0})
    value={'scope':'six historical observed development tasks;not independent accuracy or live inference',
        'source_bank_sha256':source_digest,'tasks':tasks,'new_predictions_or_labels':0}
    path=OUT/'task_bank.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False)
    print('Prepared6 matched task displays from SHA-matching raw images;no inference;no held-out input used',flush=True)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
