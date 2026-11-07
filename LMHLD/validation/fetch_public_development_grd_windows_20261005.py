"""Archive original GRD samples/schema; explicitly not terrain-corrected model input."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys,math
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import requests,numpy as np,rasterio
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rasterio.transform import GCPTransformer
from rasterio.windows import Window
from rasterio.control import GroundControlPoint
from model_registry import file_sha256
from mine_data_protocol import verify_assets,registry,reserved_case_ids
DATA=ROOT/'datasets/sar_development_pairs_20261005'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def run():
    verify_assets();inventory=registry();discovery=load(DATA/'discovery.json');plan=load(DATA/'source_plan.json')
    assert not set(discovery['selected'])&set(reserved_case_ids(inventory))
    session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=(429,500,502,503,504))))
    def sign(url):
        response=session.get('https://planetarycomputer.microsoft.com/api/sas/v1/sign',params={'href':url},timeout=(10,35));response.raise_for_status();return response.json()['href']
    for case,chosen in discovery['selected'].items():
        folder=DATA/case;selection=load(Path(chosen['pair_selection']));assert file_sha256(chosen['pair_selection'])==chosen['sha256']
        ledger=folder/'raw_acquisition.json';record=load(ledger) if ledger.exists() else {'case_id':case,'epochs':{},'errors':[],
              'source_samples_role':'unregistered rawGRD amplitude;not radarphase,coherence,mm displacement or true10mresolution',
              'model_input_admitted':False,'terrain_correction_performed':False,'detector_predictions':0}
        for period,item in selection['original_selected_items'].items():
            if period in record['epochs']:
                for path,digest in record['epochs'][period]['frozen_assets'].items():assert file_sha256(folder/path)==digest
                continue
            try:
                epoch=folder/period;epoch.mkdir(exist_ok=True);assets={};schemas={}
                for key in ['safe-manifest','schema-product-vv','schema-product-vh','schema-calibration-vv','schema-calibration-vh','schema-noise-vv','schema-noise-vh']:
                    if key not in item['assets']:raise ValueError('Required original radar schema missing:'+key)
                    url=item['assets'][key]['href'];path=epoch/(key+'.xml')
                    if not path.exists():
                        response=session.get(sign(url),timeout=(10,40));response.raise_for_status();assert len(response.content)<8*1024**2
                        import xml.etree.ElementTree as ET
                        ET.fromstring(response.content);path.write_bytes(response.content)
                    schemas[key]={'original_url':url,'sha256':file_sha256(path),'bytes':path.stat().st_size};assets[str(path.relative_to(folder))]=file_sha256(path)
                windows={};bbox=plan['cases'][case]['bbox_wgs84']
                for polarization in ['vv','vh']:
                    url=item['assets'][polarization]['href'];path=epoch/(polarization+'_original_raw_window.tif')
                    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_VERSION='1.1',GDAL_HTTP_TIMEOUT='40',GDAL_HTTP_MAX_RETRY='2',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tiff,.tif'):
                        with rasterio.open(sign(url)) as src:
                            gcps,crs=src.gcps
                            if not gcps or str(crs)!='EPSG:4326':raise ValueError('Declared WGS84 original GRD GCPs required;no guessed affine')
                            xs=[bbox[0],bbox[2],bbox[0],bbox[2]];ys=[bbox[1],bbox[1],bbox[3],bbox[3]]
                            with GCPTransformer(gcps) as transformer:rows,cols=transformer.rowcol(xs,ys)
                            left=int(max(0,min(cols)-160));top=int(max(0,min(rows)-160));right=int(min(src.width,max(cols)+161));bottom=int(min(src.height,max(rows)+161))
                            window=Window(left,top,right-left,bottom-top)
                            if max(window.width,window.height)>1800:raise ValueError('Bounded original radar window exceeded')
                            raw=src.read(1,window=window)
                            if raw.nbytes>64*1024**2:raise ValueError('Raw source window budget exceeded')
                            if not path.exists():
                                with rasterio.open(path,'w',driver='GTiff',height=raw.shape[0],width=raw.shape[1],count=1,
                                                   dtype=raw.dtype,nodata=src.nodata,compress='deflate') as dst:
                                    dst.write(raw,1)
                                    dst.gcps=([GroundControlPoint(row=p.row-top,col=p.col-left,x=p.x,y=p.y,z=p.z,id=p.id,info=p.info) for p in gcps],crs)
                                    dst.update_tags(source_scene=item['id'],data_role='development_uncorrected_GRD_source',
                                        source_original_window=json.dumps([left,top,right-left,bottom-top]),
                                        sample_semantics='detected_amplitude_DN_not_phase_or_calibratedbackscatter',
                                        geolocation='sourceGRDGCPs not independently validated terraincorrectedcoordinates',
                                        original_samples_unchanged='true',model_input_admitted='false')
                            else:
                                with rasterio.open(path) as saved:assert np.array_equal(saved.read(1),raw)
                                # An earlier failed JSON tag write can leave a
                                # complete raw/GCP file; repair metadata only.
                                with rasterio.open(path,'r+') as saved:
                                    saved.update_tags(source_scene=item['id'],data_role='development_uncorrected_GRD_source',
                                        source_original_window=json.dumps([left,top,right-left,bottom-top]),
                                        sample_semantics='detected_amplitude_DN_not_phase_or_calibratedbackscatter',
                                        geolocation='sourceGRDGCPs not independently validated terraincorrectedcoordinates',
                                        original_samples_unchanged='true',model_input_admitted='false')
                            windows[polarization]={'original_url':url,'window_original_pixel_edges':[left,top,right-left,bottom-top],
                                'source_dimensions':[src.height,src.width],'raw_shape':list(raw.shape),'source_dtype':str(raw.dtype),
                                'GCP_count':len(gcps),'GCP_CRS':str(crs),'source_resolution_range_m':item['properties']['sar:resolution_range'],
                                'source_resolution_azimuth_m':item['properties']['sar:resolution_azimuth'],
                                'native_pixel_spacing_m':[item['properties']['sar:pixel_spacing_range'],item['properties']['sar:pixel_spacing_azimuth']],
                                'sha256':file_sha256(path),'raw_samples_not_remapped_or_calibrated':True}
                            assets[str(path.relative_to(folder))]=file_sha256(path)
                record['epochs'][period]={'scene_id':item['id'],'datetime':item['properties']['datetime'],'schemas':schemas,'windows':windows,'frozen_assets':assets}
                write(ledger,record);print(case,period,'archived rawVV/VH+7schemas;not yet modelinput',flush=True)
            except Exception as exc:
                record['errors'].append({'period':period,'type':type(exc).__name__,'reason':str(exc),'model_input_admitted':False});write(ledger,record)
                print(case,period,'pending original source:',type(exc).__name__,flush=True)
    completed={case:len(load(DATA/case/'raw_acquisition.json')['epochs']) for case in discovery['selected']}
    write(DATA/'raw_acquisition_summary.json',{'completed_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
          'raw_epochs_acquired':completed,'radiometric_or_terrain_or_independent_geometry_QA_passed':False,
          'source_native_range_azimuth_resolution_20_22m_not10m_newgroundobservations':True,'new_model_inputs_admitted':0,
          'CN_XJ2023_sameperiod_missing_in_two_catalogues':True,'sources_kept_on_D':True})
    print('Raw radar source epoch counts',completed,';no radar modeladmission yet',flush=True)

if __name__=='__main__':run()
