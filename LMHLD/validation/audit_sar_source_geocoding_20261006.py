"""Check range-Doppler against unused ESA geolocation metadata, not field surveys."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import os,sys,json
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
os.environ['PROJ_USER_WRITABLE_DIRECTORY']=str(ROOT/'cache/runtime/proj');os.environ['PROJ_NETWORK']='OFF'
import numpy as np,xarray as xr
from pyproj import Transformer
import sarsen.geocoding
from terrain_correct_archived_grd_20261006 import ArchivedGRD
from model_registry import file_sha256
DATA=ROOT/'datasets/sar_development_pairs_20261005'

def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    protocol={'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'code_sha256':file_sha256(__file__),
       'case_ids':['Amyntaio','LasCruces','Hpakant'],'reference':'original ESA productgeolocationGridPoints neverusedfor orbit polynomialfit',
       'reference_is_metadata_not_independent_currentpit_groundtruth':True,
       'checks':'WGS84ellipsoid geolocationpointheight->ECEF->standard zeroDoppler+originalrangeconversion->originalimageindices',
       'median_max_original_px':1.5,'p90_max_original_px':3,'outlier_fraction_above3px_max':.1,
       'no_offset_fitting_or_warping_to_make_pass':True,'not_phase_or_velocity_accuracy':True,
       'model_admission_requires_additional_currentgeometry_andshadowreview':True}
    p=DATA/'source_geocoding_audit_protocol.json'
    if p.exists():
        old=json.loads(p.read_text(encoding='utf-8'))
        if old.get('code_sha256')!=protocol['code_sha256']:
            assert not (DATA/'source_geocoding_audit.json').exists()
            backup=DATA/'source_geocoding_audit_protocol_before_interface_fix.json'
            if not backup.exists():backup.write_bytes(p.read_bytes())
            corrected=dict(old);corrected['code_sha256']=protocol['code_sha256']
            assert all(corrected.get(k)==v for k,v in protocol.items() if k!='registered_at');write(p,corrected)
        else:assert all(old.get(k)==v for k,v in protocol.items() if k!='registered_at')
    else:write(p,protocol)
    transform=Transformer.from_crs('EPSG:4979','EPSG:4978',always_xy=True);rows=[]
    for case in protocol['case_ids']:
        for period in ['pre','post']:
            product=ArchivedGRD(case,period,'vv');points=product.annotation.findall('.//geolocationGridPoint')
            longitude=np.asarray([float(x.findtext('longitude')) for x in points]);latitude=np.asarray([float(x.findtext('latitude')) for x in points]);height=np.asarray([float(x.findtext('height')) for x in points])
            x,y,z=transform.transform(longitude,latitude,height);ecef=xr.DataArray(np.stack([x,y,z]),dims=['axis','point'],coords={'axis':[0,1,2],'point':np.arange(len(points))})
            original_times=np.asarray([np.datetime64(x.findtext('azimuthTime'),'ns') for x in points])
            interpolator=product.orbit_interpolator()
            guess=xr.DataArray((original_times-interpolator.epoch)/np.timedelta64(1,'s'),dims=['point'])
            acquisition=sarsen.geocoding.backward_geocode(ecef,interpolator,orbit_time_guess=guess,dim='axis')
            slant_range=np.sqrt((acquisition.dem_distance**2).sum('axis'));slant_time=2*slant_range/299792458.
            ground=product.slant_range_time_to_ground_range(acquisition.azimuth_time,slant_time)
            first=np.datetime64(product.info['product_first_line_utc_time'],'ns')
            line=np.asarray((acquisition.azimuth_time-first)/np.timedelta64(1,'ns'))/1e9/product.info['azimuth_time_interval']
            pixel=np.asarray(ground)/product.info['range_pixel_spacing']
            ref_line=np.asarray([float(x.findtext('line')) for x in points]);ref_pixel=np.asarray([float(x.findtext('pixel')) for x in points])
            good=np.isfinite(line)&np.isfinite(pixel);error=np.hypot(line-ref_line,pixel-ref_pixel)
            # End-point range coefficient interpolation can legitimately be outside
            # metadata support; missing controls are disclosed, never silently success.
            passed=bool(good.mean()>=.9 and np.median(error[good])<=1.5 and np.percentile(error[good],90)<=3 and (error[good]<=3).mean()>=.9)
            row={'case':case,'period':period,'metadata_control_count':len(points),'finite_controls':int(good.sum()),
                 'original_image_index_median_error_px':float(np.median(error[good])),'original_image_index_p90_error_px':float(np.percentile(error[good],90)),
                 'metadata_check_pass':passed,'median_error_original_line_px':float(np.median(line[good]-ref_line[good])),
                 'median_error_original_range_px':float(np.median(pixel[good]-ref_pixel[good])),
                 'orbit_polynomial_source_position_max_error_m':product.orbit_position_fit_max_error_m,
                 'controls_not_true_field_surveys_or_pit_currentheight':True,'offset_adjusted':False}
            rows.append(row);print(case,period,'sourceindex median/p90',row['original_image_index_median_error_px'],row['original_image_index_p90_error_px'],'pass',passed,flush=True)
    write(DATA/'source_geocoding_audit.json',{'completed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'protocol_sha256':file_sha256(p),
          'rows':rows,'metadata_geocoding_allpass':all(x['metadata_check_pass'] for x in rows),
          'independent_currentpit_ground_QA_or_model_admission':False})

if __name__=='__main__':run()
