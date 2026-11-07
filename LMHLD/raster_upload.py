# -*- coding: utf-8 -*-
"""Read multispectral TIFF without losing bands, validity or geographic metadata."""
import math
import re
import hashlib
import numpy as np


def is_tiff(raw):
    return raw[:4] in (b'II*\x00',b'MM\x00*',b'II+\x00',b'MM\x00+')


def _role(value):
    value=re.sub(r'[^A-Z0-9]','',str(value or '').upper())
    aliases={'B02':'B','B2':'B','BLUE':'B','B03':'G','B3':'G','GREEN':'G',
             'B04':'R','B4':'R','RED':'R','B08':'N','B8':'N','B8A':'N','NIR':'N','NEARINFRARED':'N'}
    return aliases.get(value)


def _require_optical_semantics(descriptions, metadata):
    """Known radar bands/units cannot be overridden by an optical band order.

    Unlabelled images remain unverified; appearance alone does not identify SAR.
    """
    for value in descriptions:
        compact=re.sub(r'[^A-Z0-9]','',str(value or '').upper())
        if compact in ('VV','VH','HH','HV') or re.match(r'^(?:VV|VH|HH|HV)(?:SIGMA|GAMMA|BETA|AMPLITUDE|INTENSITY|DB)',compact):
            raise ValueError('该栅格包含SAR雷达极化波段，当前光学模型不能处理；请上传可见光或真实BGRN影像')
    values=[str(v) for group in metadata for v in group.values()]+[str(v or '') for v in descriptions]
    for value in values:
        upper=value.upper();compact=re.sub(r'[^A-Z0-9]','',upper)
        radar_source=(re.search(r'\bSAR\b',re.sub(r'[^A-Z0-9]',' ',upper)) or
                      re.search(r'S1[ABCD]_(?:IW|EW|SM|WV|S[1-6])_(?:GRD|SLC)',upper) or
                      'SENTINEL1' in compact or 'SYNTHETICAPERTURERADAR' in compact)
        backscatter=('BACKSCATTER' in compact or re.search(r'(?:SIGMA|GAMMA|BETA)(?:NOUGHT|NAUGHT|0)',compact))
        if radar_source or backscatter:
            raise ValueError('该栅格标注为SAR雷达幅度或后向散射数据，当前光学模型不能处理；不能声明为RGB/NIR替代输入')


def decode_tiff(raw,order,max_pixels):
    try:
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.enums import ColorInterp
    except ImportError:
        raise ValueError('GeoTIFF读取依赖不可用，请配置Rasterio后重试') from None
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
            with MemoryFile(raw) as memory, memory.open() as src:
                if src.driver!='GTiff' or src.width*src.height>max_pixels or min(src.width,src.height)<2 or not 1<=src.count<=32:
                    raise ValueError('栅格类型、波段数或尺寸不在允许范围内')
                descriptions=list(src.descriptions)
                tags=src.tags()
                _require_optical_semantics(descriptions,[tags,src.tags(ns='IMAGERY')]+[src.tags(i) for i in range(1,src.count+1)])
                roles=[_role(d) for d in descriptions]
                mapping={role:i+1 for i,role in enumerate(roles) if role is not None}
                if len(mapping)!=sum(r is not None for r in roles): raise ValueError('波段描述重复或不明确')
                measured=False; verified='user_declaration'
                if all(r in mapping for r in ('B','G','R','N')):
                    indexes=[mapping[r] for r in ('B','G','R','N')]; measured=True; verified='raster_band_descriptions'
                    if order!='AUTO':
                        expected=[_role(d) for d in descriptions]
                        if src.count!=4 or expected!=list('BGRN' if order=='BGRN' else 'RGBN'):
                            raise ValueError('用户声明的波段顺序与GeoTIFF波段描述不一致，请使用自动识别')
                elif order in ('BGRN','RGBN'):
                    if src.count!=4 or src.colorinterp[3]==ColorInterp.alpha:
                        raise ValueError('真实近红外须为4个光谱波段；GeoTIFF透明度不能声明为NIR')
                    indexes=[1,2,3,4] if order=='BGRN' else [3,2,1,4]; measured=True
                elif all(r in mapping for r in ('B','G','R')):
                    indexes=[mapping[r] for r in ('B','G','R')]; verified='raster_band_descriptions'
                elif 'N' in mapping:
                    raise ValueError('该GeoTIFF缺少完整可见光波段，显示用假彩色不能作为BGRN输入')
                elif src.count==1:
                    indexes=[1,1,1]; verified='grayscale'
                elif src.count in (3,4) and all(c in src.colorinterp for c in (ColorInterp.red,ColorInterp.green,ColorInterp.blue)):
                    indexes=[src.colorinterp.index(c)+1 for c in (ColorInterp.blue,ColorInterp.green,ColorInterp.red)]; verified='raster_colour_interpretation'
                elif src.count in (3,4):
                    if src.count==4: raise ValueError('未标注4波段TIFF不能自动识别，请补波段描述或声明BGRN/RGBN')
                    # TIFF RGB storage order, unlike OpenCV's decoded BGR arrays.
                    indexes=[3,2,1]; verified='unlabelled_visible_RGB_assumption'
                else:
                    raise ValueError('多波段GeoTIFF缺少明确B/G/R/NIR描述，不能猜测波段')
                bands=src.read(indexes).astype(np.float32)
                valid=np.all(src.read_masks(indexes)>0,axis=0)&np.all(np.isfinite(bands),axis=0)
                # GTiff NoData can shadow GDAL's alpha mask; test alpha explicitly.
                alpha_indexes=[i+1 for i,c in enumerate(src.colorinterp) if c==ColorInterp.alpha]
                for alpha_index in alpha_indexes:
                    alpha=src.read(alpha_index)
                    valid &= np.isfinite(alpha)&(alpha>0)
                quality={'source':'not_provided','weather_excluded_pixels':0}
                scl_indexes=[i+1 for i,d in enumerate(descriptions) if str(d or '').upper() in ('SCL','SCENE_CLASSIFICATION')]
                if len(scl_indexes)>1:raise ValueError('SCL质量波段重复，不能识别观测质量')
                if scl_indexes:
                    scl=src.read(scl_indexes[0])
                    if not np.isfinite(scl).all() or np.any((scl<0)|(scl>11)|(scl!=np.floor(scl))):
                        raise ValueError('SCL质量波段须为0到11的原始整数类别，不能插值为小数')
                    obscured=np.isin(scl,[0,1,3,8,9,10,11])
                    quality={'source':'provided_Sentinel2_SCL','weather_excluded_pixels':int((valid&obscured).sum()),
                             'excluded_classes':[0,1,3,8,9,10,11],
                             'class_counts':{str(i):int((scl==i).sum()) for i in range(12)},
                             'native_quality_resolution_verified':False,'resampling_must_be_nearest':True,
                             'quality_labels_independently_verified':False,
                             'source_documentation':'https://documentation.dataspace.copernicus.eu/Data/Sentinel2.html'}
                    valid &= ~obscured
                scales=[float(src.scales[i-1]) for i in indexes]; offsets=[float(src.offsets[i-1]) for i in indexes]
                if not np.isfinite(scales+offsets).all() or any(v==0 for v in scales): raise ValueError('栅格辐射定标系数无效')
                if measured and src.colorinterp[indexes[3]-1]==ColorInterp.alpha:
                    raise ValueError('透明度通道不能作为真实近红外波段')
                if not measured:
                    bands=np.concatenate([bands,bands[2:3]],axis=0); scales.append(scales[2]); offsets.append(offsets[2])
                image=np.ascontiguousarray(bands.transpose(1,2,0)); image[~valid]=0
                if not valid.any(): raise ValueError('GeoTIFF没有可用观测像素')
                affine=src.transform; geo=None; mpp=None
                if src.crs and not affine.is_identity:
                    geo={'crs':str(src.crs),'transform':list(affine)[:6],'width':src.width,'height':src.height,
                         'coordinate_source':'uploaded_raster_header','independent_ground_accuracy_verified':False}
                    if src.crs.is_projected:
                        _,factor=src.crs.linear_units_factor
                        dx=math.hypot(affine.a,affine.d)*factor; dy=math.hypot(affine.b,affine.e)*factor
                        orthogonal=abs(affine.a*affine.b+affine.d*affine.e)<1e-6*max(dx*dy,1)
                        geo.update(pixel_size_m=[dx,dy],pixel_area_m2=abs(affine.a*affine.e-affine.b*affine.d)*factor**2)
                        if orthogonal and math.isclose(dx,dy,rel_tol=1e-6): mpp=dx
                    geo['meters_per_pixel']=mpp
                meta={'channel_order':'BGRN','input_channel_order':order,'nir_source':'measured' if measured else 'red_proxy',
                    'spectral_indices_available':measured,'nir_verified_by':verified,'format':'GeoTIFF' if geo else 'TIFF',
                    'source_band_descriptions':descriptions,'selected_band_indexes':indexes,'source_dtypes':list(src.dtypes),
                    'spectral_scales':scales,'spectral_offsets':offsets,'georeference':geo,
                    'observed_pixels':int(valid.sum()),'nodata_pixels':int((~valid).sum()),'_valid_mask':valid,
                    'observation_quality':quality,
                    'acquisition_datetime':tags.get('acquisition_datetime'),
                    'input_sha256':hashlib.sha256(raw).hexdigest(),
                    'declared_case_id':tags.get('case_id'),'declared_data_role':tags.get('role'),
                    'note':('真实NIR按文件波段描述识别' if measured and verified=='raster_band_descriptions' else
                            'NIR由用户声明；需确认光谱与辐射定标' if measured else '缺少真实NIR，复制R仅供检测，NDVI/NDWI不可用')+
                           '；地理标签来源于文件，不代表独立地面定位核验'}
                explicit=any(v!=1 for v in scales) or any(v!=0 for v in offsets)
                meta['radiometry_status']='provided_raster_scale_offset' if explicit else 'unknown'
                if measured and not explicit:
                    from spectral_calibration import known_calibration
                    recovered=known_calibration(raw)
                    if recovered:meta.update(recovered)
                meta['spectral_indices_available']=bool(measured and meta['radiometry_status']!='unknown')
                if measured and not meta['spectral_indices_available']:
                    meta['note']+='；缺少可核对辐射定标，光谱指数暂不可用'
                return image,meta
    except ValueError:
        raise
    except Exception:
        raise ValueError('GeoTIFF解码或地理标签无效，请检查原始栅格文件') from None


def public_band_meta(meta): return {k:v for k,v in meta.items() if not k.startswith('_')}


def observed_mask(meta,shape):
    return np.asarray(meta.get('_valid_mask',np.ones(shape,bool)),bool)


def resolve_mpp(requested,meta):
    geo=meta.get('georeference') or {}; measured=geo.get('meters_per_pixel')
    if requested is not None and geo.get('pixel_size_m') and measured is None:
        raise ValueError('栅格像元非等边或含斜切，不能用单一米/像素覆盖文件地理尺度')
    if measured is not None:
        if requested is not None and not math.isclose(float(requested),float(measured),rel_tol=.01):
            raise ValueError('请求分辨率与GeoTIFF坐标参考不一致，请使用文件中的米/像素值')
        return float(measured),'raster_header'
    return requested,'user' if requested is not None else None


def calibrated_spectral_image(image,meta):
    scales=np.array(meta.get('spectral_scales',[1]*4),np.float32); offsets=np.array(meta.get('spectral_offsets',[0]*4),np.float32)
    return image*scales+offsets
