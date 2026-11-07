"""Fixed height scenarios and historic local geometry, without fitting labels."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,os,json
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
os.environ['PROJ_USER_WRITABLE_DIRECTORY']=str(ROOT/'cache/runtime/proj');os.environ['PROJ_NETWORK']='OFF'
import numpy as np,rasterio,xarray as xr
from rasterio.warp import reproject,Resampling
from pyproj import Transformer
import sarsen.geocoding
from terrain_correct_archived_grd_20261006 import ArchivedGRD
from sar_geometry_quality import equivalent_ground_shift,historical_local_geometry
from model_registry import file_sha256
DATA=ROOT/'datasets/sar_development_pairs_20261005'
OUT=DATA/'historic_geometry_quality_v1'
OPTICAL=ROOT/'experiments/cause_terrain_round_20261005'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def stats(a,mask):
    a=np.asarray(a)[mask&np.isfinite(a)]
    return {'valid_cells':int(a.size),'median':float(np.median(a)) if a.size else None,
            'p90':float(np.percentile(a,90)) if a.size else None,'maximum':float(a.max()) if a.size else None}

def geocode(product,longitude,latitude,height):
    transform=Transformer.from_crs('EPSG:4979','EPSG:4978',always_xy=True)
    x,y,z=transform.transform(longitude,latitude,height)
    ecef=xr.DataArray(np.stack([x,y,z]),dims=['axis','y','x'],coords={'axis':[0,1,2]})
    a=sarsen.geocoding.backward_geocode(ecef,product.orbit_interpolator())
    distance=np.asarray(a.dem_distance.transpose('axis','y','x'))
    norm=np.sqrt((distance**2).sum(0));ground=product.slant_range_time_to_ground_range(a.azimuth_time,2*a.dem_distance.pipe(lambda v:np.sqrt((v*v).sum('axis')))/299792458.)
    first=np.datetime64(product.info['product_first_line_utc_time'],'ns')
    line=np.asarray((a.azimuth_time-first)/np.timedelta64(1,'ns'))/1e9/product.info['azimuth_time_interval']
    pixel=np.asarray(ground)/product.info['range_pixel_spacing']
    # Sarsen dem_distance is satellite->ground; reverse for ground->satellite.
    los=-distance/norm
    lon=np.radians(longitude);lat=np.radians(latitude)
    east=-np.sin(lon)*los[0]+np.cos(lon)*los[1]
    north=-np.sin(lat)*np.cos(lon)*los[0]-np.sin(lat)*np.sin(lon)*los[1]+np.cos(lat)*los[2]
    up=np.cos(lat)*np.cos(lon)*los[0]+np.cos(lat)*np.sin(lon)*los[1]+np.sin(lat)*los[2]
    if np.nanmedian(up)<=0:raise ValueError('Unexpected LOS sign; stop before geometry interpretation')
    return line,pixel,(east,north,up)

def save_float(path,a,profile,meaning):
    profile=dict(profile);profile.update(dtype='float32',count=1,nodata=-9999,compress='deflate')
    a=np.asarray(a,np.float32).copy();a[~np.isfinite(a)]=-9999
    with rasterio.open(path,'w',**profile) as dst:
        dst.write(a,1);dst.update_tags(meaning=meaning,not_measured_current_height_or_ground_error='true',model_input_admitted='false')

def run():
    OUT.mkdir(exist_ok=True)
    protocol={'registered_at':now(),'code_sha256':file_sha256(__file__),'geometry_code_sha256':file_sha256(ROOT/'sar_geometry_quality.py'),
        'original_geocoding_implementation_sha256':file_sha256(ROOT/'validation/terrain_correct_archived_grd_20261006.py'),
        'cases':['Amyntaio','LasCruces','Hpakant'],'height_scenarios_m':[-50.,50.],
        'height_scope':'fixed sensitivity scenarios around historical2011-2015DSM;not measured currentpit or asserted error bound',
        'source_geocoding':'actual original orbit,zeroDoppler,rangeconversion;fixed worldXY under height perturbation',
        'equivalent_ground_offset':'first-order inverse local ground-to-source Jacobian;not surveyed location error',
        'grid_posting_m':30,'jacobian_condition_max':20,'unstable_or_unobserved_left_unknown':True,
        'geometry_screen':'historic active layover and local backface;no complete passive shadow or currentpit visibility certification',
        'model_predictions':0,'model_inputs_admitted':0,'reserved_data_opened':0,'original_sources_or_reference_labels_changed':False}
    pp=OUT/'protocol.json'
    if pp.exists():assert all(load(pp).get(k)==v for k,v in protocol.items() if k!='registered_at')
    else:write(pp,protocol)
    rows=[];source_plan=load(OPTICAL/'round_plan.json')['all_sources']
    for case in protocol['cases']:
        folder=OUT/case;folder.mkdir(exist_ok=True)
        terrain=DATA/case/'terrain_correction_v1/historical_DSM_30m_WGS84_ellipsoid.tif'
        with rasterio.open(terrain) as src:
            height=src.read(1).astype(float);valid=src.read_masks(1)>0;profile=src.profile.copy();crs=src.crs;t=src.transform
        height[~valid]=np.nan
        yy,xx=np.indices(height.shape);gx=t.c+(xx+.5)*t.a;gy=t.f+(yy+.5)*t.e
        longitude,latitude=Transformer.from_crs(crs,'EPSG:4326',always_xy=True).transform(gx,gy)
        with rasterio.open(Path(source_plan[case])/'post_BGRN_10m.tif') as src:
            regions={};reference=np.load(OPTICAL/'inputs'/case/'old_reference.npy')
            for name,code in [('known_positive_core',1),('weak_normal_reference',0)]:
                mapped=np.zeros(height.shape,np.uint8)
                reproject((reference==code).astype(np.uint8),mapped,src_transform=src.transform,src_crs=src.crs,dst_transform=t,dst_crs=crs,resampling=Resampling.nearest)
                regions[name]=mapped>0
        regions['all_historic_DSM_cells']=valid
        pair={}
        for period in ['pre','post']:
            product=ArchivedGRD(case,period,'vv');line,pixel,los=geocode(product,longitude,latitude,height)
            left,top,width,h=product.details['window_original_pixel_edges']
            inside=valid&np.isfinite(line)&np.isfinite(pixel)&(pixel>=left)&(pixel<left+width-1)&(line>=top)&(line<top+h-1)
            geometry=historical_local_geometry(height,*los,t.a,t.e)
            row={'case':case,'period':period,'height_source_sha256':file_sha256(terrain),
                'inside_original_cropped_GRD_fraction':float(inside[valid].mean()),
                'historic_geometry_only':True,'complete_shadow_or_currentpit_visibility_verified':False,'scenarios':[]}
            save_float(folder/(period+'_historical_local_incidence_deg.tif'),geometry['local_incidence_deg'],profile,'local incidence from historical DSM;not currentpit survey')
            risk=np.full(height.shape,255,np.uint8);known=inside&geometry['observed'];risk[known]=0
            risk[known&geometry['active_layover_risk']]|=1;risk[known&geometry['local_backface_risk']]|=2
            p=dict(profile);p.update(dtype='uint8',count=1,nodata=255,compress='deflate')
            with rasterio.open(folder/(period+'_historical_local_risk.tif'),'w',**p) as dst:
                dst.write(risk,1);dst.update_tags(bits='1=active_layover;2=local_backface;255=unknown;zero does not certify absence of all shadows or currentpit risks',model_input_admitted='false')
            row['region_historic_geometry']={name:{'reference_cells':int(mask.sum()),'observed_reference_cells':int((mask&known).sum()),
                'active_layover_cells':int((mask&known&geometry['active_layover_risk']).sum()),'local_backface_cells':int((mask&known&geometry['local_backface_risk']).sum()),
                'local_incidence_deg':stats(geometry['local_incidence_deg'],mask&known)} for name,mask in regions.items()}
            shifts={}
            for delta in protocol['height_scenarios_m']:
                shifted_line,shifted_pixel,_=geocode(product,longitude,latitude,height+delta)
                dl=shifted_line-line;dp=shifted_pixel-pixel
                dx,dy,stable,condition=equivalent_ground_shift(line,pixel,dl,dp,t.a,t.e,20)
                supported=inside&stable;distance=np.hypot(dx,dy);distance[~supported]=np.nan
                tag='minus50m' if delta<0 else 'plus50m'
                save_float(folder/(period+'_'+tag+'_equivalent_ground_offset_m.tif'),distance,profile,'hypothetical height sensitivity;first order equivalent ground offset;not measured location error')
                row['scenarios'].append({'height_delta_m':delta,'source_index_offset_px':stats(np.hypot(dl,dp),inside),
                    'equivalent_ground_offset_m_by_region':{name:stats(distance,mask) for name,mask in regions.items()},
                    'unstable_or_missing_Jacobian_fraction':float((inside&~stable).sum()/max(1,inside.sum()))})
                shifts[delta]=(dx,dy,supported)
            pair[period]=shifts;rows.append(row)
            print(case,period,'50m-height scenario equivalent ground median',round(row['scenarios'][1]['equivalent_ground_offset_m_by_region']['all_historic_DSM_cells']['median'],2),'m;not measured error',flush=True)
        relative={}
        for delta in protocol['height_scenarios_m']:
            x1,y1,v1=pair['pre'][delta];x2,y2,v2=pair['post'][delta]
            relative[str(delta)]={name:stats(np.hypot(x2-x1,y2-y1),mask&v1&v2) for name,mask in regions.items()}
        write(folder/'paired_same_height_scenario.json',{'same_height_bias_both_epochs_equivalent_offset_difference_m':relative,
              'same_bias_may_cancel_between_similar_orbits_not_absolute_accuracy':True,
              'actual_collapse_height_change_between_epochs_not_measured_and_not_assumed_same':True})
    write(OUT/'results.json',{'completed_at':now(),'protocol_sha256':file_sha256(pp),'rows':rows,'model_inputs_admitted':0,
        'currentpit_height_unknown':True,'all_shadows_or_passive_layover_not_certified':True,'new_measured_terrain_or_survey_error_claim':False})

if __name__=='__main__':run()
