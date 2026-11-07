"""Real radar rasters must fail before any optical model is loaded."""
from pathlib import Path
import sys,unittest,base64
from unittest.mock import patch
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from upload_utils import decode_upload
import detector_service as service

def fixture(descriptions,tags=None,band_tags=None):
    values=np.arange(16*16,dtype=np.float32).reshape(16,16)+1
    with MemoryFile() as mem:
        with mem.open(driver='GTiff',width=16,height=16,count=len(descriptions),dtype='float32',crs='EPSG:32629',transform=from_origin(754950,4156920,10,10)) as dst:
            dst.write(np.stack([values+i for i in range(len(descriptions))]));dst.descriptions=descriptions
            if tags:dst.update_tags(**tags)
            if band_tags:dst.update_tags(1,**band_tags)
        return base64.b64encode(mem.read()).decode()

class RadarOpticalGuardTests(unittest.TestCase):
    def test_actual_raw_calibrated_and_terraincorrected_sources_all_rejected(self):
        for case in ('Amyntaio','LasCruces','Hpakant'):
            for period in ('pre','post'):
                for pol in ('vv','vh'):
                    files=[f'{period}/{pol}_original_raw_window.tif',f'radiometry_v1/{period}_{pol}_sigma0.tif',f'terrain_correction_v1/{period}_{pol}_gamma0_30m.tif']
                    for name in files:
                        path=ROOT/'datasets/sar_development_pairs_20261005'/case/name
                        with self.subTest(case=case,name=name),self.assertRaisesRegex(ValueError,'SAR'):
                            decode_upload(base64.b64encode(path.read_bytes()).decode())

    def test_manual_BGRN_RGBN_or_optical_descriptions_cannot_override_known_radar(self):
        for descriptions,tags in [(('VV','VH','HH','HV'),{}),(('B02','B03','B04','B08'),{'sensor':'Sentinel-1'}),((None,)*4,{'units':'sigma_nought_linear_power'})]:
            b64=fixture(descriptions,tags)
            for order in ('AUTO','BGRN','RGBN'):
                with self.subTest(descriptions=descriptions,order=order),self.assertRaisesRegex(ValueError,'SAR'):
                    decode_upload(b64,order)

    def test_band_level_radar_units_are_checked(self):
        with self.assertRaisesRegex(ValueError,'SAR'):
            decode_upload(fixture((None,),band_tags={'units':'Gamma0_dB'}))

    def test_single_and_comparison_endpoints_reject_before_model_loading(self):
        path=ROOT/'datasets/sar_development_pairs_20261005/LasCruces/terrain_correction_v1/pre_vv_gamma0_30m.tif'
        b64=base64.b64encode(path.read_bytes()).decode();client=service.app.test_client()
        with patch.object(service,'_load_upload_detector',side_effect=AssertionError('Radar must not load an optical model')) as loader:
            for endpoint,body in [('/api/detect_upload',{'image':b64}),('/api/compare',{'image_t1':b64,'image_t2':b64}),('/api/inspect_upload',{'image':b64})]:
                response=client.post(endpoint,json=body)
                self.assertEqual(response.status_code,400,response.get_json());self.assertIn('SAR',response.get_json()['error'])
                self.assertNotIn('risk_level',response.get_json())
            loader.assert_not_called()

    def test_optical_sentinel2_and_unlabelled_visible_grayscale_still_decode(self):
        image,meta=decode_upload(fixture(('B02','B03','B04','B08'),{'sensor':'Sentinel-2','units':'surface_reflectance'}))
        self.assertEqual(meta['nir_source'],'measured');self.assertEqual(image.shape,(16,16,4))
        _,meta=decode_upload(fixture((None,),{'sensor':'optical camera'}))
        self.assertEqual(meta['nir_verified_by'],'grayscale');self.assertEqual(meta['nir_source'],'red_proxy')

if __name__=='__main__':unittest.main()
