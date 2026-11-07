"""Denoise and calibrate archived source windows; no radar model admission yet."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,rasterio
from model_registry import file_sha256
from sentinel1_radiometry import calibrate_grd_sigma0
DATA=ROOT/'datasets/sar_development_pairs_20261005'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    results={}
    for case in ['Amyntaio','LasCruces','Hpakant']:
        folder=DATA/case;acquisition=load(folder/'raw_acquisition.json');assert len(acquisition['epochs'])==2
        out=folder/'radiometry_v1';out.mkdir(exist_ok=True);rows=[]
        for period,record in acquisition['epochs'].items():
            for name,digest in record['frozen_assets'].items():assert file_sha256(folder/name)==digest
            for polarization in ['vv','vh']:
                source=folder/period/(polarization+'_original_raw_window.tif')
                with rasterio.open(source) as src:raw=src.read(1);gcps,crs=src.gcps
                details=record['windows'][polarization]
                sigma,db,valid,calibration=calibrate_grd_sigma0(raw,details['window_original_pixel_edges'],
                    folder/period/('schema-calibration-'+polarization+'.xml'),folder/period/('schema-noise-'+polarization+'.xml'),
                    folder/period/('schema-product-'+polarization+'.xml'))
                assets={}
                for kind,array,dtype,nodata in [('sigma0',sigma,'float32',0),('sigma0_dB',db,'float32',-9999),('observed',valid.astype(np.uint8),'uint8',None)]:
                    path=out/(period+'_'+polarization+'_'+kind+'.tif')
                    if path.exists():
                        with rasterio.open(path) as saved:assert np.array_equal(saved.read(1),array)
                    else:
                        with rasterio.open(path,'w',driver='GTiff',width=raw.shape[1],height=raw.shape[0],count=1,dtype=dtype,nodata=nodata,compress='deflate',gcps=gcps,crs=crs) as dst:
                            dst.write(array,1);dst.update_tags(data_role='development_calibrated_but_not_terraincorrected_SAR',
                                units=calibration['units'],source_raw_sha256=file_sha256(source),model_input_admitted='false',geometric_ground_accuracy_verified='false')
                    assets[path.name]=file_sha256(path)
                row={'period':period,'polarization':polarization,'raw_source_sha256':file_sha256(source),'calibration':calibration,
                     'sigma0_dB_percentiles_observed':np.percentile(db[valid],[1,50,99]).tolist(),'assets':assets}
                rows.append(row);print(case,period,polarization,'sigma0calibrated valid',round(float(valid.mean()),4),flush=True)
        receipt={'processed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'case_id':case,
                 'raw_acquisition_sha256':file_sha256(folder/'raw_acquisition.json'),'calibration_code_sha256':file_sha256(ROOT/'sentinel1_radiometry.py'),
                 'rows':rows,'original_DN_images_unmodified':True,'terrain_and_independent_ground_QA_passed':False,
                 'source_range_azimuth_resolution_m':[20,22],'pixel_spacing_is_not_actual10mresolution':True,'model_input_admitted':False}
        p=out/'calibration.json'
        if p.exists():
            old=load(p);assert all(old.get(k)==v for k,v in receipt.items() if k!='processed_at')
        else:write(p,receipt)
        results[case]={'receipt':str(p),'sha256':file_sha256(p),'calibrated_polarization_epochs':len(rows)}
    write(DATA/'radiometry_summary.json',{'results':results,'new_CNN_or_accuracy_runs':0,'radar_model_admission':False,'original_sources_preserved':True})

if __name__=='__main__':run()
