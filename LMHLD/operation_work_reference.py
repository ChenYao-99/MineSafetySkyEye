"""Conservative source-backed work appearances; not accident absence or slope safety."""
import numpy as np
from rasterio.features import rasterize
from rasterio.warp import transform_geom

def source_mine_clearing_geometries(collection):
    result=[]
    for feature in collection.get('features',[]):
        p=feature.get('properties') or {}
        if (p.get('class') or p.get('descr'))=='Clearing - Mine':
            if not feature.get('geometry'):raise ValueError('Government clearing geometry missing')
            result.append((p.get('objectid'),feature['geometry']))
    return result

def derive_work_core(geometries,crs,transform,pre,post,common_visible,uncertainty_m=40):
    from shapely.geometry import shape,mapping
    if not crs.is_projected or max(abs(x-10) for x in (abs(transform.a),abs(transform.e)))>1e-6:
        raise ValueError('Projected native10m image grid required')
    pre=np.asarray(pre,np.float32);post=np.asarray(post,np.float32);visible=np.asarray(common_visible,bool)
    if pre.shape!=post.shape or pre.shape[:2]!=visible.shape or pre.shape[2]!=4:
        raise ValueError('Same native BGRN pair and visibility mask required')
    source_shapes=[];core_shapes=[]
    for identifier,geom in geometries:
        projected=shape(transform_geom('EPSG:4326',crs,geom))
        if not projected.is_valid:raise ValueError('Invalid original government polygon;no silent repair')
        source_shapes.append((mapping(projected),1))
        interior=projected.buffer(-uncertainty_m)
        if not interior.is_empty:core_shapes.append((mapping(interior),1))
    def burn(shapes):return rasterize(shapes,out_shape=visible.shape,transform=transform,fill=0,dtype='uint8').astype(bool) if shapes else np.zeros(visible.shape,bool)
    provider_area=burn(source_shapes);provider_core=burn(core_shapes)
    def ndvi(a):
        den=a[:,:,3]+a[:,:,2]
        return np.divide(a[:,:,3]-a[:,:,2],den,out=np.zeros_like(den),where=np.abs(den)>1e-6)
    before,after=ndvi(pre),ndvi(post)
    # This conjunction supports a visible mine-clearing appearance inside the
    # published annual geometry, not a exact clearing date or hidden stability.
    concur=(before>.35)&(after<.25)&((before-after)>.15)
    core=provider_core&visible&concur
    return provider_area,provider_core,core,{'published_mine_clearing_pixels':int(provider_area.sum()),
        'after40m_boundary_uncertainty_pixels':int(provider_core.sum()),
        'common_visible_published_interior_pixels':int((provider_core&visible).sum()),
        'observed_concurrent_work_appearance_pixels':int(core.sum()),
        'outside_core_unknown':True,'normal_accident_absence_or_stability_ground_truth':False,
        'exact_per_polygon_clearing_dates_verified':False}
