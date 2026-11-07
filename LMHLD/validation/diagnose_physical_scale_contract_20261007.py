"""Controlled input-resolution replay;not a new mine/accident accuracy benchmark."""
from pathlib import Path
from datetime import datetime
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,rasterio,cv2
from detector import LandslideDetector
from mine_data_protocol import assert_development_cases,verify_assets
from model_registry import ModelRegistry
OUT=ROOT/'experiments/physical_scale_contract_20261007'


def main():
    ids=['Amyntaio','LasCruces','Hpakant','CN_XJ2023']
    # Resolve registry event IDs explicitly rather than allowing an arbitrary file glob.
    from mine_data_protocol import registry
    inventory=registry();entries=[]
    for name in ids:
        matches=[key for key,case in inventory['cases'].items() if key==name or name.lower() in str(case.get('name','')).lower() or name.lower() in str(case.get('site_name','')).lower()]
        if not matches:
            aliases={'Amyntaio':'GR_AMY2017','LasCruces':'ES_LC2019','Hpakant':'MM_HP2020','CN_XJ2023':'CN_XJ2023'}
            matches=[aliases[name]]
        entries.append(matches[0])
    assert_development_cases(entries);verify_assets();OUT.mkdir(exist_ok=True)
    weights=ROOT/'models_prior/real_ft_T1b_ours_seed42/best_model.pth'
    artifact=ModelRegistry(ROOT/'model_registry.json',ROOT/'models_prior').verify('upload:mine',weights,
        {'patch_size':32,'threshold':.4,'patch_norm':'minmax','input_scale':1.0})
    protocol={'registered_at':datetime.now().astimezone().isoformat(),'development_cases':entries,
        'source_resolution_m':10,'controlled_replica_resolution_m':2.5,'same_underlying_scene_not_new_real_mines':True,
        'fixed_model':artifact,'threshold_or_model_tuning':False,'new_model_fits':0,'reserved_predictions':0,
        'comparison':'original10m vs sameinformation replicated4x at2.5m;correctedinput averages back to original10m'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    model=LandslideDetector(str(weights),patch_size=32,threshold=.4);results=[]
    folders={'Amyntaio':'Amyntaio','LasCruces':'LasCruces','Hpakant':'Hpakant','CN_XJ2023':'Xinjing'}
    for name in ids:
        path=ROOT/'datasets/mine_benchmark_20261002/development'/folders[name]/'post_BGRN_10m.tif'
        with rasterio.open(path) as src:image=src.read().transpose(1,2,0).astype(np.float32)
        replica=cv2.resize(image,(image.shape[1]*4,image.shape[0]*4),interpolation=cv2.INTER_NEAREST)
        restored=cv2.resize(replica,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_AREA)
        assert np.array_equal(image,restored)
        times=[];probabilities=[]
        for value in [image,replica,restored]:
            begin=time.monotonic();probabilities.append(model.detect_image(value,patch_norm='minmax',max_batch=128));times.append(time.monotonic()-begin)
        original=probabilities[0]>=.4;naive=probabilities[1]>=.4;corrected=probabilities[2]>=.4
        reference=cv2.resize(original.astype(np.uint8),(naive.shape[1],naive.shape[0]),interpolation=cv2.INTER_NEAREST)>0
        disagreement=float((reference!=naive).mean());union=(reference|naive).sum()
        row={'case':name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'image_shape':list(image.shape),
            'native_model_mask_fraction':float(original.mean()),'uncorrected_replica_mask_fraction':float(naive.mean()),
            'replica_disagreement_with_same_scene_reference':disagreement,'replica_reference_IoU':float((reference&naive).sum()/max(union,1)),
            'scale_corrected_mask_exactly_preserved':bool(np.array_equal(original,corrected)),
            'scale_corrected_probability_max_error':float(np.max(np.abs(probabilities[0]-probabilities[2]))),
            'seconds_native_uncorrected_corrected':times,'not_ground_truth_accuracy_measurement':True}
        results.append(row);print(name,row,flush=True)
        (OUT/'results_partial.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'results.json').write_text(json.dumps({'protocol':protocol,'cases':results,'physical_scale_mismatch_demonstrated':any(r['replica_disagreement_with_same_scene_reference']>.01 for r in results),'new_trained_models_or_reserved_predictions':0},ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
