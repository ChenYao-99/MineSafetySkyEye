"""Actual default detector on observed development work evidence; no cloud or final data."""
from pathlib import Path
import base64,json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,rasterio
from model_registry import file_sha256
import detector_service as service
DATA=ROOT/'datasets/slats_operation_development_20261005';CASE=DATA/'AU_AMRUN_SLATS2324'
OUT=ROOT/'experiments/government_work_runtime_validation_20261005';OUT.mkdir(exist_ok=True)

def run():
    acceptance=json.loads((DATA/'work_reference_acceptance.json').read_text(encoding='utf-8'))
    ref=acceptance['accepted']['AU_AMRUN_SLATS2324']
    if file_sha256(ref['reference'])!=ref['reference_sha256']:raise ValueError('Accepted reference changed')
    package=OUT/'post_original_BGRN_with_original_SCL.tif'
    with rasterio.open(CASE/'post_BGRN_10m.tif') as a,rasterio.open(CASE/'post_SCL_10m.tif') as b:
        if a.crs!=b.crs or a.transform!=b.transform or a.shape!=b.shape:raise ValueError('Quality grid does not match')
        values=a.read();quality=b.read(1);profile=a.profile.copy();tags=a.tags();scales=a.scales;offsets=a.offsets
        profile.update(count=5)
        if not package.exists():
            with rasterio.open(package,'w',**profile) as out:
                out.write(values,indexes=[1,2,3,4]);out.write(quality.astype(values.dtype),5)
                out.descriptions=('B02','B03','B04','B08','SCL');out.scales=tuple(scales)+(1.,);out.offsets=tuple(offsets)+(0.,)
                out.update_tags(**tags,original_BGRN_sha256=file_sha256(CASE/'post_BGRN_10m.tif'),
                                original_SCL_sha256=file_sha256(CASE/'post_SCL_10m.tif'),quality_native_resolution_m='20',
                                SCL_existing_nearest10m_grid_preserved='true')
        with rasterio.open(package) as out:
            np.testing.assert_array_equal(out.read([1,2,3,4]),values);np.testing.assert_array_equal(out.read(5),quality)
            assert out.crs==a.crs and out.transform==a.transform
            np.testing.assert_array_equal(out.scales[:4],scales);np.testing.assert_array_equal(out.offsets[:4],offsets)
    output=OUT/'default_mine_API_on_Amrun_yearly_work.json'
    if output.exists():
        print('Using recorded actual API probe;no repeated GPU inference');return
    loaded=service._load_upload_detector('mine')
    if loaded is None:raise ValueError('Actual registered default artifact unavailable')
    detector,patch_norm,input_scale=loaded;original=detector.detect_image;captured=[]
    def capture(*args,**kwargs):
        probability=original(*args,**kwargs);captured.append(np.asarray(probability).copy());return probability
    detector.detect_image=capture
    try:
        response=service.app.test_client().post('/api/detect_upload',json={
            'image':base64.b64encode(package.read_bytes()).decode(),'model_tag':'mine','channel_order':'auto','max_batch':16})
    finally:detector.detect_image=original
    if response.status_code!=200:raise ValueError('Actual detector rejected source packet: '+str(response.get_json()))
    result=response.get_json()
    assert len(captured)==1 and np.isfinite(captured[0]).all()
    with rasterio.open(CASE/'work_reference_v1_SLATS/expected_work_mask.tif') as src:core=src.read(1)>0
    activation=float((captured[0][core]>=result['threshold']).mean())
    np.save(OUT/'default_mine_Amrun_yearly_probability.npy',captured[0])
    result.pop('overlay_b64',None)
    receipt={'scope':'actual default API smoke on source-supported expected-work development;not new blind or verified safety false-positive rate',
        'input_packet_sha256':file_sha256(package),'original_spectral_and_SCL_samples_unchanged':True,
        'SCL_native20m_existing_nearest10m_mask_does_not_add_observations':True,
        'reference_acceptance_sha256':file_sha256(DATA/'work_reference_acceptance.json'),
        'expected_work_core_pixels':int(core.sum()),'default_expected_work_activation':activation,
        'actual_detector_result':result,'actual_native_inference_calls':1,'cloud_requests':0,
        'reserved_predictions':0,'production_weights_changed':False,
        'baseline_default_also_not_claimed_full_crossmine_accepted':True}
    output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Actual default expectedwork activation',round(activation,6),'pixels',int(core.sum()),'threshold',result['threshold'],
          'request_id',result['request_id'],'originalDN+SCL preserved',flush=True)

if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
