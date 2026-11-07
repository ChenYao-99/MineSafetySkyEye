"""Typed review-location boxes bound to immutable input evidence, never accident labels."""
import json,math,re

PREFIX='manual:'


def frame(det,report_id):
    shape=det.get('size')
    if not isinstance(shape,list) or len(shape)!=2 or any(isinstance(x,bool) or not isinstance(x,int) or x<2 for x in shape):
        raise ValueError('原始影像尺寸缺失，不能添加定位框')
    paired='t1_ratio' in (det.get('stats') or {})
    meta=(det.get('band_meta') or {}).get('t2',{}) if paired else det.get('band_meta') or {}
    if not isinstance(meta,dict) or not isinstance(meta.get('input_sha256'),str) or not re.fullmatch('[a-f0-9]{64}',meta['input_sha256']):
        raise ValueError('原始影像标识缺失，不能添加定位框')
    return {'report_id':report_id,'request_id':det.get('request_id'),'input_sha256':meta['input_sha256'],
        'image_shape':shape,'coordinate_frame':'t2_comparison_grid' if paired else 'original_image_grid',
        'acquisition_datetime':meta.get('acquisition_datetime')}


def location_box(box,det,report_id):
    source=frame(det,report_id);h,w=source['image_shape']
    if not isinstance(box,list) or len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box):
        raise ValueError('定位框须为4个有限像素坐标')
    x0,y0,x1,y1=box
    if not (0<=x0<x1<=w and 0<=y0<y1<=h) or (x1-x0)*(y1-y0)<1:
        raise ValueError('定位框须位于原图内，且面积不少于1像素')
    return {'kind':'review_location_box','box_px':[round(float(v),4) for v in box],
        'source':source,'geometry_is_accident_boundary':False,'verified_ground_truth':False,
        'physical_failure_area_m2':None,'observation_support_independently_checked':False,
        'note':'人工补记的核查位置；定位框不是事故边界，不能用框面积代替灾害面积或证明模型召回提升'}


def encode(box,note,det,report_id):
    return json.dumps({'type':'manual_review_location_v1','annotation':location_box(box,det,report_id),'review_note':note},
        ensure_ascii=False,allow_nan=False,separators=(',',':'))


def decode(event,det,report_id):
    if not event['target_key'].startswith(PREFIX):return event
    if not re.fullmatch(r'manual:[a-f0-9]{32}',event['target_key']):raise ValueError('人工定位框标识无效')
    payload=json.loads(event['note'])
    if payload.get('type')!='manual_review_location_v1' or not isinstance(payload.get('review_note'),str):
        raise ValueError('人工定位框记录格式无效')
    stored=payload['annotation'];validated=location_box(stored.get('box_px'),det,report_id)
    if stored!=validated:raise ValueError('人工定位框与原始影像标识或范围不一致')
    return {**event,'note':payload['review_note'],'manual_annotation':validated}
