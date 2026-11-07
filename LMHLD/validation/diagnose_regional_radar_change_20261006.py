"""Physical amplitude-change evidence under fixed uncertain-height scenarios.

No classifier, pixel reference correction, or production input admission.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,json
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
from audit_historical_sar_geometry_20261006 import geocode,stats
from terrain_correct_archived_grd_20261006 import ArchivedGRD
from model_registry import file_sha256
import numpy as np,rasterio
from rasterio.warp import reproject,Resampling
from pyproj import Transformer
from scipy.ndimage import uniform_filter,map_coordinates
DATA=ROOT/'datasets/sar_development_pairs_20261005'
OUT=DATA/'regional_amplitude_change_diagnostic_v1'
OPTICAL=ROOT/'experiments/cause_terrain_round_20261005'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def smooth_power(path):
    with rasterio.open(path) as src:power=src.read(1).astype(float)
    valid=np.isfinite(power)&(power>0)
    support=uniform_filter(valid.astype(float),size=7,mode='constant',cval=0)
    mean=uniform_filter(np.where(valid,power,0),size=7,mode='constant',cval=0)
    mean=np.divide(mean,support,out=np.full_like(mean,np.nan),where=support>=.95)
    return mean,support

def run():
    OUT.mkdir(exist_ok=True)
    protocol={'registered_at':now(),'code_sha256':file_sha256(__file__),
        'geocoding_code_sha256':file_sha256(ROOT/'validation/audit_historical_sar_geometry_20261006.py'),
        'cases':['Amyntaio','LasCruces','Hpakant'],'height_scenarios_m':[-50.,0.,50.],
        'scenarios_are_not_measured_current_height_or_error_bound':True,
        'source_units':'original thermallydenoised calibrated sigma0linear power;not gamma0,phase,coherence or deformation',
        'denoising_for_diagnostic_only':'fixed7x7source-index linearpower mean;>=.95 observedsupport;no zeros counted as observed',
        'sampling':'bilinear originalsource address from actual orbit+rangeDoppler;all 4 surrounding pixels observed',
        'magnitude_statistic':'0.5*(abs(VVlogratio_dB)+abs(VHlogratio_dB))',
        'fixed_strong_amplitude_change_diagnostic_db':3.,'not_detector_or_optimized_decision_threshold':True,
        'no_per_case_or_global_bias_offset_fitted':True,'regions':'frozen knownpositivecore and weaknormal references;not full independent polygons',
        'analysis_ground_grid_posting_m':30,'effective_spatial_support_coarser_than_nativeSAR_notnew10mresolution':True,
        'model_inputs_admitted':0,'models_trained_or_prediction_calls':0,'reserved_inputs':0,'originalsources_and_references_unchanged':True}
    pp=OUT/'protocol.json'
    if pp.exists():assert all(load(pp).get(k)==v for k,v in protocol.items() if k!='registered_at')
    else:write(pp,protocol)
    rows=[];source_plan=load(OPTICAL/'round_plan.json')['all_sources']
    for case in protocol['cases']:
        folder=OUT/case;folder.mkdir(exist_ok=True)
        with rasterio.open(DATA/case/'terrain_correction_v1/historical_DSM_30m_WGS84_ellipsoid.tif') as src:
            height=src.read(1).astype(float);valid=src.read_masks(1)>0;profile=src.profile.copy();crs=src.crs;t=src.transform
        height[~valid]=np.nan
        yy,xx=np.indices(height.shape);longitude,latitude=Transformer.from_crs(crs,'EPSG:4326',always_xy=True).transform(t.c+(xx+.5)*t.a,t.f+(yy+.5)*t.e)
        reference=np.load(OPTICAL/'inputs'/case/'old_reference.npy');regions={}
        with rasterio.open(Path(source_plan[case])/'post_BGRN_10m.tif') as src:
            for name,code in [('known_positive_core',1),('weak_normal_reference',0)]:
                a=np.zeros(height.shape,np.uint8);reproject((reference==code).astype(np.uint8),a,src_transform=src.transform,src_crs=src.crs,dst_transform=t,dst_crs=crs,resampling=Resampling.nearest);regions[name]=a>0
        region_union=regions['known_positive_core']|regions['weak_normal_reference']
        samples={};assets={}
        for period in ['pre','post']:
            product=ArchivedGRD(case,period,'vv');left,top,w,h=product.details['window_original_pixel_edges']
            prepared={}
            for pol in ['vv','vh']:
                path=DATA/case/'radiometry_v1'/(period+'_'+pol+'_sigma0.tif');prepared[pol]=smooth_power(path);assets[str(path)]=file_sha256(path)
                other=load(DATA/case/'raw_acquisition.json')['epochs'][period]['windows'][pol]['window_original_pixel_edges']
                assert [left,top,w,h]==other
            for delta in protocol['height_scenarios_m']:
                line,pixel,_=geocode(product,longitude,latitude,height+delta);coordinates=np.stack([line-top,pixel-left]);inside=valid&(line>=top)&(line<top+h-1)&(pixel>=left)&(pixel<left+w-1)
                values={}
                for pol,(power,support) in prepared.items():
                    available=np.isfinite(power)
                    a=map_coordinates(np.where(available,power,0),coordinates,order=1,mode='constant',cval=0,prefilter=False)
                    observed=map_coordinates(available.astype(float),coordinates,order=1,mode='constant',cval=0,prefilter=False)>=1-1e-6
                    good=inside&observed&(a>0);db=np.full(height.shape,np.nan);db[good]=10*np.log10(a[good]);values[pol]=db
                samples[(period,delta)]=values
        change={}
        row={'case':case,'calibrated_source_sha256':assets,'weaknormal_is_not_certified_accidentfree_work':True,'scenarios':[]}
        for delta in protocol['height_scenarios_m']:
            vv=samples[('post',delta)]['vv']-samples[('pre',delta)]['vv'];vh=samples[('post',delta)]['vh']-samples[('pre',delta)]['vh']
            magnitude=(abs(vv)+abs(vh))*.5;change[delta]=magnitude
            entry={'hypothetical_height_delta_m':delta,'regions':{}}
            for name,mask in regions.items():
                good=mask&np.isfinite(magnitude)
                entry['regions'][name]={'reference_cells':int(mask.sum()),'observed_cells':int(good.sum()),
                    'VV_logratio_db':stats(vv,mask),'VH_logratio_db':stats(vh,mask),'mean_absolute_logratio_db':stats(magnitude,mask),
                    'fraction_above_fixed_3db_diagnostic':float((magnitude[good]>=3).mean()) if good.any() else None}
            row['scenarios'].append(entry)
            np.savez_compressed(folder/('change_'+str(int(delta))+'m.npz'),VV_logratio_db=vv.astype(np.float32),VH_logratio_db=vh.astype(np.float32),mean_absolute_logratio_db=magnitude.astype(np.float32))
        array=np.stack([change[d] for d in protocol['height_scenarios_m']]);all_observed=np.isfinite(array).all(0)
        spread=np.full(height.shape,np.nan);spread[all_observed]=array[:,all_observed].max(0)-array[:,all_observed].min(0)
        robust=np.zeros(height.shape,bool);robust[all_observed]=(array[:,all_observed]>=3).all(0)
        row['scenario_sensitivity']={name:{'change_magnitude_spread_db':stats(spread,mask),
            'all3_scenarios_observed_cells':int((mask&all_observed).sum()),'all3_scenarios_above_fixed3db_fraction':float(robust[mask&all_observed].mean()) if (mask&all_observed).any() else None} for name,mask in regions.items()}
        rows.append(row)
        core=row['scenarios'][1]['regions']['known_positive_core'];normal=row['scenarios'][1]['regions']['weak_normal_reference']
        print(case,'noheightshift meanabsolutechange median core/weaknormal',round(core['mean_absolute_logratio_db']['median'],3),round(normal['mean_absolute_logratio_db']['median'],3),'dB;not recall/falsealarm result',flush=True)
    write(OUT/'results.json',{'completed_at':now(),'protocol_sha256':file_sha256(pp),'rows':rows,'model_inputs_admitted':0,
        'new_CNNfits_or_classifiers':0,'no_crossmine_accuracy_or_currentpit_groundtruth_claim':True,
        'CN_XJ2023_missingSAR_not_evaluated_or_filled':True})

if __name__=='__main__':run()
