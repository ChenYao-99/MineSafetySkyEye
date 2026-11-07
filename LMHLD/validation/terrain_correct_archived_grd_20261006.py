"""Sarsen terrain correction of actual cropped GRD, with a stated historic DSM."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import os,sys,json,functools
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
os.environ['PROJ_USER_WRITABLE_DIRECTORY']=str(ROOT/'cache/runtime/proj')
os.environ['PROJ_NETWORK']='OFF'
import numpy as np,rasterio,xarray as xr
from rasterio.warp import reproject,Resampling
import sarsen,sarsen.orbit,sarsen.datamodel,sarsen.sentinel1,xarray_sentinel
from sentinel1_radiometry import xml_root,interpolated_range_vectors
from model_registry import file_sha256
DATA=ROOT/'datasets/sar_development_pairs_20261005'
TERRAIN=ROOT/'datasets/historical_terrain_development_20261005'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

class ArchivedGRD(sarsen.datamodel.GroundRangeSarProduct):
    def __init__(self,case,period,polarization):
        self.folder=DATA/case;self.period=period;self.polarization=polarization
        self.record=load(self.folder/'raw_acquisition.json')['epochs'][period]
        self.details=self.record['windows'][polarization]
        self.annotation=xml_root(self.folder/period/('schema-product-'+polarization+'.xml'))
        self.calibration=xml_root(self.folder/period/('schema-calibration-'+polarization+'.xml'))
        self.info={
            'product_type':'GRD','range_pixel_spacing':float(self.annotation.findtext('.//rangePixelSpacing')),
            'azimuth_pixel_spacing':float(self.annotation.findtext('.//azimuthPixelSpacing')),
            'azimuth_time_interval':float(self.annotation.findtext('.//azimuthTimeInterval')),
            'product_first_line_utc_time':self.annotation.findtext('.//productFirstLineUtcTime'),
            'image_slant_range_time':float(self.annotation.findtext('.//imageInformation/slantRangeTime')),
            'incidence_angle_mid_swath':float(self.annotation.findtext('.//incidenceAngleMidSwath')),
        }
        self.case=case

    @property
    def product_type(self):return 'GRD'

    @functools.cached_property
    def calibrated_beta(self):
        path=self.folder/'radiometry_v1'/(self.period+'_'+self.polarization+'_sigma0.tif')
        with rasterio.open(path) as src:sigma=src.read(1)
        left,top,width,height=self.details['window_original_pixel_edges'];rows=np.arange(top,top+height,dtype=float);columns=np.arange(left,left+width,dtype=float)
        nodes=self.calibration.findall('.//calibrationVector')
        a=interpolated_range_vectors(nodes,'sigmaNought',rows,columns);b=interpolated_range_vectors(nodes,'betaNought',rows,columns)
        beta=sigma.astype(np.float64)*a*a/(b*b)
        beta[sigma<=0]=np.nan
        times=np.datetime64(self.info['product_first_line_utc_time'],'ns')+np.rint(rows*self.info['azimuth_time_interval']*1e9).astype('timedelta64[ns]')
        return xr.DataArray(beta.astype(np.float32),dims=['azimuth_time','ground_range'],
            coords={'azimuth_time':times,'ground_range':columns*self.info['range_pixel_spacing']},
            attrs={'units':'denoised_beta_nought_linear_power','source':'actualoriginalGRD croppedwindow calibrated using originalsigma/beta LUTs'})

    def beta_nought(self):return self.calibrated_beta

    def geospatial_bounds(self):
        b=load(DATA/'source_plan.json')['cases'][self.case]['bbox_wgs84']
        return f'POLYGON(({b[0]} {b[1]},{b[2]} {b[1]},{b[2]} {b[3]},{b[0]} {b[3]},{b[0]} {b[1]}))'

    @functools.cached_property
    def interpolator(self):
        nodes=self.annotation.findall('.//orbit')
        times=np.asarray([np.datetime64(x.findtext('time'),'ns') for x in nodes]);positions=np.asarray([[float(x.findtext('position/'+axis)) for axis in ['x','y','z']] for x in nodes])
        data=xr.DataArray(positions,dims=['azimuth_time','axis'],coords={'azimuth_time':times,'axis':[0,1,2]})
        result=sarsen.orbit.OrbitPolyfitInterpolator.from_position(data,deg=5)
        predicted=np.asarray(result.position(xr.DataArray(times,dims=['azimuth_time'])))
        if predicted.shape!=positions.shape:predicted=predicted.T
        error=np.linalg.norm(predicted-positions,axis=1)
        if not np.isfinite(error).all() or error.max()>1:raise ValueError('Original orbit interpolation residual exceeds1m')
        self.orbit_position_fit_max_error_m=float(error.max())
        return result

    def orbit_interpolator(self,**kwargs):return self.interpolator

    def state_vectors(self):
        nodes=self.annotation.findall('.//orbit')
        times=np.asarray([np.datetime64(x.findtext('time'),'ns') for x in nodes])
        positions=np.asarray([[float(x.findtext('position/'+axis)) for axis in ['x','y','z']] for x in nodes])
        return xr.DataArray(positions,dims=['azimuth_time','axis'],coords={'azimuth_time':times,'axis':[0,1,2]})

    @functools.cached_property
    def conversion(self):
        nodes=[x for x in self.annotation.findall('.//coordinateConversion') if x.findtext('azimuthTime')]
        if not nodes:raise ValueError('Original range conversion coefficients absent')
        times=np.asarray([np.datetime64(x.findtext('azimuthTime'),'ns') for x in nodes])
        coefficients=[np.fromstring(x.findtext('srgrCoefficients'),sep=' ') for x in nodes]
        if len({len(x) for x in coefficients})!=1:raise ValueError('Inconsistent coordinate polynomial degree')
        return xr.Dataset({'sr0':('azimuth_time',[float(x.findtext('sr0')) for x in nodes]),
            'srgrCoefficients':(('azimuth_time','degree'),np.stack(coefficients))},
            coords={'azimuth_time':times,'degree':np.arange(len(coefficients[0]))})

    def slant_range_time_to_ground_range(self,azimuth_time,slant_range_time):
        return xarray_sentinel.slant_range_time_to_ground_range(azimuth_time,slant_range_time,self.conversion)

    def grid_parameters(self,grouping_area_factor=(3.,3.)):
        return sarsen.sentinel1.azimuth_slant_range_grid(self.info,grouping_area_factor)

def ellipsoidal_dem(case):
    source=TERRAIN/case/'historical_DSM_projected_30m.tif';geoid=DATA/case/'original_EGM2008_geoid_native_window.tif'
    out=DATA/case/'terrain_correction_v1';out.mkdir(exist_ok=True);target=out/'historical_DSM_30m_WGS84_ellipsoid.tif'
    with rasterio.open(source) as src:
        z=src.read(1).astype(np.float32);valid=src.read_masks(1)>0;profile=src.profile.copy();transform=src.transform;crs=src.crs
    with rasterio.open(geoid) as src:
        offset=np.full(z.shape,np.nan,np.float32);reproject(src.read(1),offset,src_transform=src.transform,src_crs=src.crs,dst_transform=transform,dst_crs=crs,resampling=Resampling.bilinear)
    if not np.isfinite(offset[valid]).all():raise ValueError('No verified geoid support for verticalconversion')
    converted=z+offset;converted[~valid]=-9999
    if target.exists():
        with rasterio.open(target) as saved:assert np.array_equal(saved.read(1),converted)
    else:
        profile.update(count=1,dtype='float32',nodata=-9999,compress='deflate')
        with rasterio.open(target,'w',**profile) as dst:
            dst.write(converted,1);dst.update_tags(vertical_reference='WGS84ellipsoid h=EGM2008orthometricH+originalgeoidN',
                source_original_height_sha256=file_sha256(source),source_geoid_sha256=file_sha256(geoid),
                date_scope='historical2011-2015DSM;notcurrentpit survey or deformation',native_terrain_resolution_m='30')
    return target,float(np.median(offset[valid]))

def run():
    processing={'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'code_sha256':file_sha256(__file__),
        'sarsen_version':sarsen.__version__,'xarray_sentinel_version':xarray_sentinel.__version__,
        'cases':['Amyntaio','LasCruces','Hpakant'],'dem_native_m':30,'historical_DSM_period':'2011-2015;not currentpit geometry',
        'vertical_conversion':'actualoriginalEGM2008geoidwindow added toorthometricheight toWGS84ellipsoid',
        'correct_radiometry':'gamma_nearest','grouping_area_factor':[3,3],'output_posting_m':30,
        'no_more_independent_resolution_than_native20by22m_orDEM30m':True,
        'source_model_admission_requires_layover_shadow_and_independent_geometric_QA':True,
        'no_phase_or_coherence_or_mm_displacement_claim':True,'CN_missing_exactdate_pair_not_imputed':True,
        'model_inputs_admitted':0,'original_groundtruth_masks_modified':False}
    p=DATA/'terrain_correction_protocol.json'
    if p.exists():
        old=load(p)
        if old.get('code_sha256')!=processing['code_sha256']:
            assert not any(DATA.glob('*/terrain_correction_v1/*_gamma0_30m.tif')), 'Never change code after a successful terrain output'
            backup=DATA/'terrain_correction_protocol_before_adapter_fix.json'
            if not backup.exists():backup.write_bytes(p.read_bytes())
            corrected=dict(old);corrected['code_sha256']=processing['code_sha256']
            assert all(corrected.get(k)==v for k,v in processing.items() if k!='registered_at')
            write(p,corrected)
        else:assert all(old.get(k)==v for k,v in processing.items() if k!='registered_at')
    else:write(p,processing)
    results={}
    for case in processing['cases']:
        target,offset=ellipsoidal_dem(case);folder=target.parent;entries=[]
        for period in ['pre','post']:
            for polarization in ['vv','vh']:
                product=ArchivedGRD(case,period,polarization);path=folder/(period+'_'+polarization+'_gamma0_30m.tif');simulation=folder/(period+'_'+polarization+'_simulated_beta_30m.tif')
                if not path.exists():
                    sarsen.terrain_correction(product,str(target),output_urlpath=str(path),simulated_urlpath=str(simulation),
                        correct_radiometry='gamma_nearest',interp_method='nearest',grouping_area_factor=(3.,3.),
                        chunks=256,radiometry_chunks=256,radiometry_bound=32,enable_dask_distributed=False)
                with rasterio.open(path) as src:a=src.read(1);valid=np.isfinite(a)&(a>0)
                if valid.mean()<.6:raise ValueError('Terraincorrected sourcecoverage below.6')
                entries.append({'period':period,'polarization':polarization,'path':str(path),'sha256':file_sha256(path),
                    'simulated_source_geometric_area_sha256':file_sha256(simulation),'valid_fraction':float(valid.mean()),
                    'geoid_offset_median_m':offset,'model_input_admitted':False,'independent_currentpit_geometry_verified':False})
                print(case,period,polarization,'terraincorrected historicDSM coverage',round(float(valid.mean()),3),flush=True)
        receipt={'entries':entries,'protocol_sha256':file_sha256(p),'original_sources_unmodified':True,
                 'currentpit_layover_shadow_and_ground_QA_not_yetpassed':True,'radar_model_admitted':False}
        write(folder/'processing.json',receipt);results[case]=str(folder/'processing.json')
    write(DATA/'terrain_correction_summary.json',{'processed':results,'radar_model_admitted':False,'new_accuracy_or_mm_grounddeformation_claim':False})

if __name__=='__main__':run()
