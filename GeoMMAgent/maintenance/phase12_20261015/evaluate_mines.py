"""Same service preprocessing/grid, four previously seen development mines only."""
import sys
import time
from pathlib import Path
from run_experiment import (ROOT,HERE,OUT,BASE,read,write,sha,now,RGBTaskAdapter,torch,
                            LandslideDetector,evaluate)
import numpy as np
import rasterio
sys.path.insert(0,str(ROOT/'validation'))
from mine_data_protocol import assert_development_cases
from nir_synth_utils import load_synth
from rgb_adapter import load_guarded_adapter


def raster(path):
    with rasterio.open(path) as src:return src.read(1),src.transform,src.crs


def main():
    protocol=read(OUT/'protocol.json')
    assert_development_cases(protocol['mine_regression_cases'])
    for row in [protocol['base_model'],*protocol['protected_registries']]:
        if sha(row['path'])!=row['sha256']:raise ValueError('Frozen runtime or protected registry changed')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    final=OUT/'mine_results.json'
    if final.exists():
        print('Existing completed mine evaluation retained; no repeated selection',flush=True)
        return
    records=[]
    mine_data=[]
    for mine in protocol['mine_regression_cases']:
        folder=ROOT/'datasets/mine_benchmark_20261002/development'/('Xinjing' if mine=='CN_XJ2023' else mine)
        image_path=folder/'post_BGRN_10m.tif'
        with rasterio.open(image_path) as src:
            image=src.read().transpose(1,2,0).astype(np.float32)
            grid,crs=src.transform,src.crs
        label_path=folder/'reference_v1_20261004/labels.tif' if mine=='CN_XJ2023' else folder/'partial_labels.tif'
        normal_path=folder/'reference_v1_20261004/normal_mask.tif' if mine=='CN_XJ2023' else ROOT/'datasets/mine_benchmark_20261002/review_v2_20261003'/mine/'reviewed_normal_mask.tif'
        labels,lg,lc=raster(label_path)
        normal,ng,nc=raster(normal_path)
        if labels.shape!=image.shape[:2] or normal.shape!=labels.shape or grid!=lg or grid!=ng or crs!=lc or crs!=nc:
            raise ValueError('Native image/reference grids must match; no metric-time warp')
        if not set(np.unique(labels)).issubset({0,1,255}):raise ValueError('Unknown label schema')
        proxy=image.copy();proxy[:,:,3]=proxy[:,:,2]
        mine_data.append((mine,proxy,labels,normal>0))
        records.append({'mine':mine,'image_path':str(image_path),'image_sha256':sha(image_path),
                        'label_path':str(label_path),'label_sha256':sha(label_path),
                        'normal_path':str(normal_path),'normal_sha256':sha(normal_path),
                        'role':'previously seen development regression; no mine pixels used in this adapter training'})
    evaluation_plan={'registered_at':now(),'protocol_sha256':sha(OUT/'protocol.json'),
          'grid':'exact LandslideDetector.detect_image stride32, minmax, max_batch128, source native10m',
          'cases':records,'baselines':['red_copy','legacy_nir_mse','existing_rgb_guarded'],
          'thresholds':protocol['thresholds'],'no_recalibration_or_per_mine_threshold':True,
          'negatives_are_project_reviewed_weak_references_not_field_safety_truth':True,
          'unannotated_unknown_pixels_scored':False,'new_unopened_test_predictions':0,
          'Noto_event_remains_qualification_only':True}
    write(OUT/'mine_evaluation_plan.json',evaluation_plan)
    detector=LandslideDetector(str(BASE),threshold=.4)
    results=[]
    legacy=None
    def score(name,seed=None,synth=None,current_detector=None):
        current_detector=current_detector or detector
        group=[]
        for mine,image,labels,normal in mine_data:
            begin=time.monotonic()
            probabilities=current_detector.detect_image(image,patch_norm='minmax',max_batch=128,nir_synth_model=synth)
            values={}
            for threshold in protocol['thresholds']:
                hit=probabilities>=threshold
                pos,bg=labels==1,labels==0
                values[str(threshold)]={
                    'known_positive_recall':float(hit[pos].mean()),
                    'old_weak_background_activation':float(hit[bg].mean()) if bg.any() else None,
                    'reviewed_normal_activation':float(hit[normal].mean()) if normal.any() else None,
                    'known_positive_pixels':int(pos.sum()),'weak_background_pixels':int(bg.sum()),
                    'reviewed_normal_pixels':int(normal.sum()),'unknown_pixels_scored':0}
            group.append({'mine':mine,'thresholds':values,'seconds':time.monotonic()-begin})
            print(name,seed,mine,values['0.4'],flush=True)
        result={'arm':name,'seed':seed,'mines':group}
        results.append(result)
        write(OUT/'mine_results_partial.json',{'results':results})
    score('red_copy')
    legacy=load_synth(dev='cuda')
    score('legacy_nir_mse',synth=legacy)
    pack=np.load(protocol['pack_path'])
    # All legacy comparisons use the same validation rows and fixed threshold.
    class LatentWrapped(torch.nn.Module):
        def __init__(self,m):super().__init__();self.m=m
        def forward(self,x):return self.m(x).clamp(0,1)
    legacy_source=evaluate(detector.model,pack,LatentWrapped(legacy))
    del legacy;torch.cuda.empty_cache()
    guarded=load_guarded_adapter(ROOT/'models_prior/rgb_guarded_20261001/adapter.json',ROOT/'models_prior',device='cuda')
    score('existing_rgb_guarded',current_detector=guarded)
    del guarded;torch.cuda.empty_cache()
    trainings=read(OUT/'training_results.json')['fits']
    for result in trainings:
        if sha(result['weights'])!=result['weights_sha256']:raise ValueError('Fixed-final adapter changed')
        adapter=RGBTaskAdapter().cuda().eval()
        adapter.load_state_dict(torch.load(result['weights'],map_location='cuda',weights_only=True))
        score(result['arm'],result['seed'],synth=adapter)
        del adapter;torch.cuda.empty_cache()
    write(final,{'completed_at':now(),'evaluation_plan_sha256':sha(OUT/'mine_evaluation_plan.json'),
          'results':results,'legacy_nir_source_validation':legacy_source,
          'new_unopened_mine_predictions':0,'Noto_predictions':0,'independent_blind_claim':False,
          'same_threshold_comparison_note':'RGB guarded legacy service threshold .5; this comparison reports both .4 and .5 for all arms without choosing a new threshold'})


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
