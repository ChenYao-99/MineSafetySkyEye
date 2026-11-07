"""Current real development inputs only; no training, downloads or reserve inference."""
from pathlib import Path
import os,sys,json,base64,time,hashlib
ROOT=Path(r'D:\GeoMMAgent\maintenance\slim_project');L=Path(r'D:\LMHLD')
sys.path.insert(0,str(L));os.environ['YOLO_AUTOINSTALL']='false'
import cv2,numpy as np,torch
import detector_service as service
client=service.app.test_client();records=[]
def call(url,payload):
 started=time.perf_counter();response=client.post(url,json=payload)
 value=response.get_json()
 assert response.status_code==200,(url,response.status_code,value.get('error'))
 assert not value.get('error'),value.get('error')
 return value,round(time.perf_counter()-started,3)
inputs=Path(r'D:\GeoMMAgent\cache\case_delivery_20261008_v2\inputs')
encoded={epoch:base64.b64encode((inputs/(epoch+'_BGRN_10m.tif')).read_bytes()).decode() for epoch in ['pre','post']}
single,elapsed=call('/api/detect_upload',{'image':encoded['post'],'model_tag':'mine','channel_order':'BGRN'})
assert single['threshold']==.4 and single['band_meta']['nir_source']=='measured'
assert single['scale_meta']['meters_per_pixel']==10 and single['quant']['ndvi']['available']
assert single['model_info']['artifact']['weights_sha256']=='0d8d8a0540419d4409951ed91e69499fac206b68303f3feec08a32f0841cebef'
assert abs(single['stats']['area_ratio']-.0871)<.0002
records.append({'task':'native_single','elapsed_seconds':elapsed,'ratio':single['stats']['area_ratio'],'NDVI_available':True,'weights_sha256':single['model_info']['artifact']['weights_sha256']})
rgb=cv2.imdecode(np.frombuffer(base64.b64decode(single['rgb_b64']),np.uint8),cv2.IMREAD_COLOR)[:96,:96]
small=base64.b64encode(cv2.imencode('.png',rgb)[1]).decode()
blend,elapsed=call('/api/detect_upload',{'image':small,'model_tag':'rgb_guarded'})
assert blend['model_info']['adapter_weight']==.15 and blend['threshold']==.5
assert not blend['quant']['ndvi']['available']
records.append({'task':'RGB_adapter','elapsed_seconds':elapsed,'adapter_weight':.15,'physical_NDVI':False})
synthetic,elapsed=call('/api/detect_upload',{'image':small,'model_tag':'mine','nir_synth':True})
assert synthetic['nir_synth'] and not synthetic['quant']['ndvi']['available']
records.append({'task':'optional_NIR_synthesis','elapsed_seconds':elapsed,'physical_NDVI':False})
paired,elapsed=call('/api/compare',{'image_t1':encoded['pre'],'image_t2':encoded['post'],'model_tag':'mine','channel_order':'BGRN'})
assert abs(paired['stats']['net_ratio']-.0005)<.0002
assert paired['band_meta']['t1']['nir_source']=='measured' and paired['band_meta']['t2']['nir_source']=='measured'
records.append({'task':'native_pair','elapsed_seconds':elapsed,'t1_ratio':paired['stats']['t1_ratio'],'t2_ratio':paired['stats']['t2_ratio'],'net_ratio':paired['stats']['net_ratio']})
for region,fuse in [('Emilia-Romagna_Italy_32',False),('Hokkaido_Japan_128',True)]:
 result,elapsed=call('/api/detect',{'region':region,'num':1,'start':0,'fuse':fuse,'ds_reject':fuse})
 assert len(result['samples'])==1 and result['fused']==fuse
 records.append({'task':'source_dataset_and_fusion' if fuse else 'source_dataset','region':region,'elapsed_seconds':elapsed,'fused':result['fused'],'samples':1})
out={'passed':True,'records':records,'GPU':torch.cuda.get_device_name(0),'reserve_predictions':0,'cloud_calls':0,'new_training':0,'scope':'post-cleanup functionality, not new accuracy claims'}
(ROOT/'gpu_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(out,ensure_ascii=False))
