"""Correct the same real GRD onto the registered expanded background grid."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,json
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'validation'))
from terrain_correct_archived_grd_20261006 import ArchivedGRD
from model_registry import file_sha256
import sarsen,rasterio,numpy as np
OUT=ROOT/'datasets/sar_development_pairs_20261005/LasCruces/terrain_background_v2'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
    acquisition=load(OUT/'acquisition.json')
    for name,digest in acquisition['assets_sha256'].items():assert file_sha256(OUT/name)==digest
    dem=OUT/'historical_DSM_30m_WGS84_ellipsoid.tif'
    protocol={'registered_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'code_sha256':file_sha256(__file__),
       'source_adapter_sha256':file_sha256(ROOT/'validation/terrain_correct_archived_grd_20261006.py'),
       'expanded_terrain_acquisition_sha256':file_sha256(OUT/'acquisition.json'),'terrain_sha256':file_sha256(dem),
       'same_original_dates_and_raw_GRD_windows':True,'case':'LasCruces','polarizations':['vv','vh'],
       'correct_radiometry':'gamma_nearest','grouping_area_factor':[3,3],'interpolation':'nearest','analysis_posting_m':30,
       'original_raw_samples_unchanged':True,'model_admitted':False,'background_features_not_true_fieldsurvey':True}
    pp=OUT/'processing_protocol.json'
    if pp.exists():assert all(load(pp).get(k)==v for k,v in protocol.items() if k!='registered_at')
    else:write(pp,protocol)
    rows=[]
    for period in ['pre','post']:
        for polarization in protocol['polarizations']:
            product=ArchivedGRD('LasCruces',period,polarization);out=OUT/(period+'_'+polarization+'_gamma0_30m.tif');sim=OUT/(period+'_'+polarization+'_simulated_beta_30m.tif')
            if not out.exists():sarsen.terrain_correction(product,str(dem),output_urlpath=str(out),simulated_urlpath=str(sim),correct_radiometry='gamma_nearest',interp_method='nearest',grouping_area_factor=(3.,3.),chunks=256,radiometry_chunks=256,radiometry_bound=32,enable_dask_distributed=False)
            with rasterio.open(out) as src:a=src.read(1);valid=np.isfinite(a)&(a>0)
            rows.append({'period':period,'polarization':polarization,'path':str(out),'sha256':file_sha256(out),'valid_fraction':float(valid.mean()),'model_admitted':False})
            print('LasCruces expanded',period,polarization,'original GRD support',round(float(valid.mean()),3),flush=True)
    write(OUT/'processing.json',{'protocol_sha256':file_sha256(pp),'rows':rows,'model_admitted':False,'partial_source_coverage_not_imputed':True})

if __name__=='__main__':run()
