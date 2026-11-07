"""Reproduce the fixed missing-observation check and real local API acceptance.

Only four explicit DEVELOPMENT cases. No cloud calls, archive writes, fitting,
reserved-case access, new data downloads or changes to historical receipts.
"""
from pathlib import Path
import sys,json,hashlib,base64
import numpy as np
ROOT=Path('D:/LMHLD');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import detector_service as service
from input_quality import prepare_inference_context
from mine_data_protocol import assert_development_cases


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def missing_tiff(path,left):
    import rasterio
    from rasterio.io import MemoryFile
    with rasterio.open(path) as src: data=src.read();profile=src.profile.copy()
    h,w=data.shape[1:];width=max(8,w//8);scl=np.full((h,w),5,np.uint16)
    if left:scl[:,:width]=9
    else:scl[:,-width:]=9
    profile.update(driver='GTiff',count=5,dtype='uint16',nodata=0)
    with MemoryFile() as memory:
        with memory.open(**profile) as dataset:
            dataset.write(np.concatenate([data,scl[None]],axis=0))
            dataset.descriptions=('B02','B03','B04','B08','SCL')
            dataset.scales=(.0001,)*4+(1.,);dataset.offsets=(0.,)*5
        return base64.b64encode(memory.read()).decode()


def run():
    import torch
    protocol=json.loads((HERE/'context_protocol.json').read_text(encoding='utf-8'))
    original=json.loads((HERE/'context_sensitivity.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'input_quality.py')==protocol['helper_sha256']
    assert_development_cases(protocol['cases'])
    det,norm,scale=service._load_upload_detector('mine');rows=[]
    for filename,digest in protocol['source_images'].items():
        path=Path(filename);assert sha(path)==digest
        case=path.parent.name if path.parent.name!='Xinjing' else 'CN_XJ2023'
        image,meta=service._prepare_upload(base64.b64encode(path.read_bytes()).decode(),'AUTO',True)
        valid=service.observed_mask(meta,image.shape[:2]);reference=det.detect_image(image*scale,patch_norm=norm,max_batch=64)
        h,w=valid.shape;_,xx=np.indices(valid.shape);stripes=(xx%64<24)|(xx%64>=32)
        rectangles=np.ones(valid.shape,bool)
        rectangles[h//4:h//4+max(8,h//10),w//3:w//3+max(8,w//10)]=False
        rectangles[2*h//3:2*h//3+max(8,h//12),2*w//3:2*w//3+max(8,w//12)]=False
        for name,pattern in [('stripes',stripes),('rectangles',rectangles)]:
            support=valid&pattern;metrics={}
            for method in ('zero_common','nearest_observed'):
                model_image,_=prepare_inference_context(image,support,method)
                assert np.array_equal(model_image[support],image[support])
                prediction=det.detect_image(model_image*scale,patch_norm=norm,max_batch=64)
                metrics[method]={'mean_absolute_probability_error':float(np.abs(prediction[support]-reference[support]).mean()),
                    'mask_flip_rate':float(((prediction[support]>=det.threshold)!=(reference[support]>=det.threshold)).mean()),'evaluated_observed_pixels':int(support.sum())}
            saved=next(r for r in original['rows'] if r['case']==case and r['pattern']==name)
            for method in metrics:
                for k in ('mean_absolute_probability_error','mask_flip_rate'):
                    if abs(metrics[method][k]-saved['metrics'][method][k])>2e-6:raise ValueError('Missing-context sensitivity replay differs')
            rows.append({'case':case,'pattern':name,'replayed':True})
    assert det.artifact_info['weights_sha256']==original['registered_weight_sha256']
    folder=ROOT/'datasets/mine_benchmark_20261002/development/LasCruces';post=folder/'post_BGRN_10m.tif'
    encode=lambda p:base64.b64encode(p.read_bytes()).decode();client=service.app.test_client()
    previous=json.loads((HERE.parent/'review_cleanup_20261104/gpu_and_performance.json').read_text(encoding='utf-8'))['real_development_gpu']
    api=[]
    for name,url,payload in [('single','/api/detect_upload',{'image':encode(post)}),('roi','/api/detect_upload',{'image':encode(post),'roi':[.2,.2,.7,.7]}),
                             ('pair','/api/compare',{'image_t1':encode(folder/'pre_BGRN_10m.tif'),'image_t2':encode(post)})]:
        response=client.post(url,json=payload);r=response.get_json();assert response.status_code==200,r.get('error')
        current={k:v for k,v in r['stats'].items() if k not in ('regions','region_areas')}
        if current!=previous[name]['stats']:raise ValueError('Fully observed real API statistics changed')
        api.append({'scenario':name,'status':200,'fully_observed_stats_unchanged':True})
    # Actual DN values, synthetic SCL missing strips: underlying common pixels
    # are identical and must yield zero apparent change with the real GPU model.
    response=client.post('/api/compare',json={'image_t1':missing_tiff(post,True),'image_t2':missing_tiff(post,False)})
    r=response.get_json();assert response.status_code==200,r.get('error');stats=r['stats']
    assert all(stats[k]==0 for k in ('n_added_regions','n_expanded_regions','n_removed_regions','net_ratio'))
    assert r['input_quality']['common_valid']['t1']['inference_context']['context_only_pixels']>0
    api.append({'scenario':'identical_measured_DN_with_different_synthetic_missing_strips','status':200,'stats':stats,'false_change':False})
    record={'status':'passed','sensitivity_replays':rows,'real_gpu_api':api,'weights_sha256':det.artifact_info['weights_sha256'],
        'runtime_sha256':service.DETECTOR_RUNTIME_SHA256,'new_reserved_predictions':0,'cloud_calls':0,'not_a_new_mine_accuracy_benchmark':True,'torch':torch.__version__}
    (HERE/'development_acceptance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    service._release_models();print(json.dumps({'sensitivity_replays':len(rows),'GPU_API_scenarios':len(api),'status':'passed'},ensure_ascii=False))


if __name__=='__main__':run()
