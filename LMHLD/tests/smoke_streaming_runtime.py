"""Real default-weight equivalence and end-to-end native raster service validation."""
import base64,json,sys,time
from pathlib import Path
import numpy as np
import rasterio
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import detector_service as service
from model_registry import file_sha256


def dense_reference(det,image):
    # Frozen old protocol: collect, per-channel minmax, predict, row-order stitch.
    h,w=image.shape[:2];ps=32
    padded=np.pad(image,((0,(-h)%ps),(0,(-w)%ps),(0,0)),mode='reflect')
    positions=[(y,x) for y in range(0,padded.shape[0],ps) for x in range(0,padded.shape[1],ps)]
    patches=np.stack([padded[y:y+ps,x:x+ps].transpose(2,0,1) for y,x in positions])
    lo=patches.min((2,3),keepdims=True);hi=patches.max((2,3),keepdims=True)
    patches=(patches-lo)/(hi-lo+1e-6)*255.0
    probabilities=np.concatenate([det.predict_batch(patches[i:i+16]) for i in range(0,len(patches),16)])
    output=np.zeros(padded.shape[:2],np.float32)
    for (y,x),prob in zip(positions,probabilities):output[y:y+ps,x:x+ps]=prob
    return output[:h,:w]


def run():
    folder=ROOT/'datasets/mine_benchmark_20261002/development/LasCruces'
    source=folder/'post_BGRN_10m.tif'
    with rasterio.open(source) as f:image=f.read().transpose(1,2,0).astype(np.float32)
    det,norm,scale=service._load_upload_detector('mine')
    measurements=[]
    for mode in ('NIR','RGBproxy'):
        x=image.copy()
        if mode=='RGBproxy':x[:,:,3]=x[:,:,2]
        expected=dense_reference(det,x)
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();actual=det.detect_image(x,patch_norm='minmax',max_batch=16)
        np.testing.assert_array_equal(expected,actual)
        delta=float(np.abs(expected-actual).max())
        measurements.append({'mode':mode,'max_probability_difference':delta,
            'mask_equal_at_04':bool(np.array_equal(expected>=.4,actual>=.4)),
            'elapsed_s':round(time.perf_counter()-start,3),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None})
    client=service.app.test_client()
    raw=base64.b64encode(source.read_bytes()).decode()
    response=client.post('/api/detect_upload',json={'image':raw,'roi':[.25,.25,.9,.9]})
    assert response.status_code==200,response.get_json()
    body=response.get_json()
    assert body['model_info']['artifact']['weights_sha256']=='0d8d8a0540419d4409951ed91e69499fac206b68303f3feec08a32f0841cebef'
    assert body['band_meta']['nir_source']=='measured'
    assert body['scale_meta']['meters_per_pixel']==10
    assert body['stats']['scope']=='roi'
    assert body['request_id']==response.headers['X-KAT-Request-ID']
    pair=client.post('/api/compare',json={'image_t1':base64.b64encode((folder/'pre_BGRN_10m.tif').read_bytes()).decode(),'image_t2':raw})
    assert pair.status_code==200,pair.get_json()
    assert pair.get_json()['registration']['raster_grids_checked']
    default_hash=body['model_info']['artifact']['weights_sha256']
    del det
    for tag in ('base','mine'):
        loaded=service._load_upload_detector(tag)
        assert loaded is not None
        assert list(service._upload_detectors)==[tag]
        assert loaded[0].artifact_info['weights_sha256']==file_sha256(service.UPLOAD_MODELS[tag][0])
        del loaded
    output=ROOT/'experiments/runtime_integrity_20261004';output.mkdir(exist_ok=True)
    record={'case':'LasCruces observed development only','default_weight_sha256':default_hash,
            'streaming_equivalence':measurements,'native_raster_upload_and_pair_passed':True,
            'model_switch_cache_limit_passed':True,'request_trace_passed':True,
            'new_reserved_predictions':0,'no_field_accuracy_or_safety_claim':True,
            'torch':torch.__version__,'rasterio':rasterio.__version__}
    (output/'real_gpu_smoke.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    service._release_models()
    print(json.dumps(record))


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
