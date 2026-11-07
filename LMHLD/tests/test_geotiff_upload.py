# -*- coding: utf-8 -*-
"""Actual GeoTIFF transport, observed-area accounting and projected polygon export."""
import base64
import json
import unittest
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import rasterio
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.enums import ColorInterp
import detector_service as service
from detector import LandslideDetector,ModelNumericalError
from upload_utils import decode_upload
from raster_upload import calibrated_spectral_image


def tiff_data(array,nodata=0,transform=None,descriptions=('B02','B03','B04','B08'),alpha=False,calibration=False):
    with MemoryFile() as mem:
        with mem.open(driver='GTiff',width=array.shape[2],height=array.shape[1],count=len(array),dtype=str(array.dtype),
                      crs='EPSG:32629',transform=transform or from_origin(754950,4156920,10,10),nodata=nodata) as dst:
            dst.write(array)
            for i,d in enumerate(descriptions,1):
                if d: dst.set_band_description(i,d)
            if alpha: dst.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
            if calibration:
                dst.scales=(.0001,)*4
                dst.offsets=(-.1,)*4 if calibration is True else (0.,)*4
        return base64.b64encode(mem.read()).decode()


class Stub(LandslideDetector):
    def __init__(self): self.threshold,self.patch_size,self.device=.4,32,'cpu'
    def detect_image(self,image,**kwargs): return (image[:,:,0]>128).astype(np.float32)


class GeoTIFFContracts(unittest.TestCase):
    def setUp(self):
        self.client=service.app.test_client(); self.stub=Stub()
        self.loader=patch.object(service,'_load_upload_detector',return_value=(self.stub,'minmax',1)); self.loader.start(); self.addCleanup(self.loader.stop)
        self.x=np.stack([np.full((64,64),v,np.uint16) for v in (100,50,30,90)])
        self.x[0,4:20,4:20]=200; self.x[:,:,32:]=0; self.b64=tiff_data(self.x,calibration='linear')

    def test_true_native_four_bands_are_not_decoded_as_grayscale(self):
        path=Path('D:/LMHLD/datasets/mine_benchmark_20261002/development/LasCruces/post_BGRN_10m.tif')
        image,meta=decode_upload(base64.b64encode(path.read_bytes()).decode(),'AUTO')
        with rasterio.open(path) as src: expected=src.read().transpose(1,2,0).astype(np.float32)
        np.testing.assert_array_equal(image,expected)
        self.assertEqual(meta['nir_source'],'measured'); self.assertEqual(meta['georeference']['meters_per_pixel'],10)

    def test_inspection_returns_preview_without_predicting(self):
        with patch.object(service,'_load_upload_detector',side_effect=AssertionError('inspection must not load a model')):
            r=self.client.post('/api/inspect_upload',json={'image':self.b64})
        self.assertEqual(r.status_code,200); d=r.get_json()
        self.assertEqual(d['image_shape'],[64,64]); self.assertEqual(d['scale_meta']['meters_per_pixel'],10)
        self.assertNotIn('_valid_mask',d['band_meta']); self.assertEqual(d['band_meta']['nodata_pixels'],2048)
        self.assertTrue(base64.b64decode(d['preview_b64']).startswith(b'\x89PNG'))

    def test_nodata_excluded_from_roi_stats_and_spectral_denominator(self):
        r=self.client.post('/api/detect_upload',json={'image':self.b64}); self.assertEqual(r.status_code,200)
        d=r.get_json(); self.assertEqual(d['stats']['total_pixels'],2048); self.assertEqual(d['stats']['area_ratio'],.125)
        self.assertEqual(d['stats']['landslide_area_m2'],25600)
        self.assertEqual(d['scale_meta']['source'],'raster_header')
        self.assertAlmostEqual(d['quant']['ndvi']['mean'],.5,places=4)
        r=self.client.post('/api/detect_upload',json={'image':self.b64,'roi':[.5,0,1,1]})
        self.assertEqual(r.status_code,400); self.assertIn('有效观测',r.get_json()['error'])

    def test_metadata_and_manual_channel_or_scale_conflict_is_rejected(self):
        for extra in ({'channel_order':'RGBN'},{'meters_per_pixel':.5}):
            r=self.client.post('/api/detect_upload',json={'image':self.b64,**extra}); self.assertEqual(r.status_code,400)

    def test_alpha_cannot_be_forced_to_measured_nir(self):
        data=np.ones((4,16,16),np.uint8)*200
        b64=tiff_data(data,descriptions=(None,)*4,alpha=True)
        _,meta=decode_upload(b64,'AUTO'); self.assertEqual(meta['nir_source'],'red_proxy')
        with self.assertRaises(ValueError): decode_upload(b64,'BGRN')

    def test_nodata_cannot_shadow_transparent_pixels(self):
        data=np.ones((4,16,16),np.uint8)*200
        data[3,:,8:]=0
        image,meta=decode_upload(tiff_data(data,nodata=0,descriptions=(None,)*4,alpha=True),'AUTO')
        self.assertEqual(meta['observed_pixels'],128)
        self.assertEqual(image[:,8:].sum(),0)

    def test_actual_scl_excludes_obscured_pixels_but_keeps_bright_bare_mine(self):
        x=np.ones((5,64,64),np.uint16)*100
        x[0,4:20,4:20]=200
        x[0,4:20,44:56]=120
        x[4]=5  # Actual SCL bare ground, including bright mine surface.
        for column,kind in enumerate((3,8,9,10,11)):
            x[4,:,column*8:(column+1)*8]=kind
        b64=tiff_data(x,descriptions=('B02','B03','B04','B08','SCL'))
        _,meta=decode_upload(b64,'AUTO')
        self.assertEqual(meta['observed_pixels'],24*64)
        self.assertEqual(meta['observation_quality']['weather_excluded_pixels'],40*64)
        response=self.client.post('/api/detect_upload',json={'image':b64})
        self.assertEqual(response.status_code,200)
        body=response.get_json()
        self.assertEqual(body['stats']['valid_pixels'],24*64)
        self.assertEqual(body['input_quality']['weather_assessment'],'provided_SCL_screening')
        self.assertEqual(body['stats']['landslide_pixels'],0)

    def test_invalid_interpolated_scl_and_all_cloud_are_rejected(self):
        x=np.ones((5,16,16),np.float32)*100
        x[4]=5.5
        with self.assertRaisesRegex(ValueError,'整数类别'):
            decode_upload(tiff_data(x,descriptions=('B02','B03','B04','B08','SCL')),'AUTO')
        x[4]=9
        response=self.client.post('/api/detect_upload',json={'image':tiff_data(x,descriptions=('B02','B03','B04','B08','SCL'))})
        self.assertEqual(response.status_code,400)
        self.assertNotIn('risk_level',response.get_json())

    def test_file_radiometric_offset_is_used_for_spectral_values(self):
        x=np.stack([np.full((16,16),v,np.uint16) for v in (2000,2000,1800,3000)])
        image,meta=decode_upload(tiff_data(x,calibration=True),'AUTO'); reflectance=calibrated_spectral_image(image,meta)
        self.assertAlmostEqual(float(reflectance[0,0,2]),.08,places=6); self.assertAlmostEqual(float(reflectance[0,0,3]),.2,places=6)

    def test_unknown_radiometry_blocks_physical_indices_and_known_original_restores_XML(self):
        response=self.client.post('/api/detect_upload',json={'image':tiff_data(self.x)})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.get_json()['band_meta']['nir_source'],'measured')
        self.assertFalse(response.get_json()['quant']['ndvi']['available'])
        self.assertNotIn('mean',response.get_json()['quant']['ndvi'])
        path=Path('D:/LMHLD/datasets/mine_benchmark_20261002/development/Xinjing/post_BGRN_10m.tif')
        image,meta=decode_upload(base64.b64encode(path.read_bytes()).decode(),'AUTO')
        self.assertEqual(meta['radiometry_status'],'verified_local_product_XML')
        self.assertEqual(meta['spectral_offsets'],[-.1]*4)
        with rasterio.open(path) as f:
            expected=f.read().transpose(1,2,0)
            valid=np.all(f.read_masks()>0,axis=0)
        expected[~valid]=0  # Existing transport contract zeroes all bands of an invalid observation.
        np.testing.assert_array_equal(image,expected)
        reflectance=calibrated_spectral_image(image,meta)
        np.testing.assert_allclose(reflectance,expected*.0001-.1,atol=1e-6)

    def test_different_geographic_grids_cannot_be_compared_as_aligned(self):
        other=tiff_data(self.x,transform=from_origin(755050,4156920,10,10))
        r=self.client.post('/api/compare',json={'image_t1':self.b64,'image_t2':other,'registration':'prealigned'})
        self.assertEqual(r.status_code,400); self.assertIn('格网',r.get_json()['error'])

    def test_same_real_raster_grid_uses_native_pixels_without_visual_warp(self):
        first=self.x.copy();first[:,:,32:]=first[:,:,:32]
        second=first.copy();second[0,4:20,4:20]=100;second[0,23:35,4:25]=200
        with patch.object(service,'align_periods',wraps=service.align_periods) as alignment:
            r=self.client.post('/api/compare',json={'image_t1':tiff_data(first),'image_t2':tiff_data(second),'registration':'auto'})
        self.assertEqual(r.status_code,200,r.get_json())
        self.assertEqual(alignment.call_args.args[2],'prealigned')
        registration=r.get_json()['registration']
        self.assertEqual(registration['basis'],'same_uploaded_raster_grid')
        self.assertFalse(registration['independent_ground_accuracy_verified'])
        self.assertNotIn('transform',registration)

    def test_projection_and_holes_survive_geojson_export(self):
        x=np.stack([np.full((64,64),v,np.uint16) for v in (100,50,30,90)])
        x[0,8:40,8:40]=200; x[0,16:24,16:24]=100
        d=self.client.post('/api/detect_upload',json={'image':tiff_data(x)}).get_json()
        r=self.client.post('/api/geometry_export',json={'regions':d['stats']['regions'],'georeference':d['band_meta']['georeference'],'format':'geojson'})
        self.assertEqual(r.status_code,200); g=json.loads(base64.b64decode(r.get_json()['content_b64']))
        f=g['features'][0]; self.assertEqual(len(f['geometry']['coordinates']),2)
        lon,lat=f['geometry']['coordinates'][0][0]; self.assertTrue(-6.2<lon<-6.0 and 37.4<lat<37.6)
        self.assertEqual(f['properties']['coordinate_source'],'uploaded_raster_header')

    def test_nonfinite_model_output_cannot_become_low_risk(self):
        with patch.object(self.stub,'detect_image',return_value=np.full((64,64),np.nan,np.float32)):
            r=self.client.post('/api/detect_upload',json={'image':self.b64})
        self.assertEqual(r.status_code,503)
        self.assertNotIn('risk_level',r.get_json())

    def test_reserved_original_and_quality_derivative_cannot_be_opened_by_upload(self):
        root=Path('D:/LMHLD/datasets/mine_benchmark_20261002/reserved_candidates/US_BC2021')
        for file in (root/'post_BGRN_10m.tif',root/'quality_audit/post_BGRN_calibrated_tags.tif'):
            image=base64.b64encode(file.read_bytes()).decode()
            with patch.object(service,'_load_upload_detector',side_effect=AssertionError('reserved data must not load a model')):
                inspection=self.client.post('/api/inspect_upload',json={'image':image})
                self.assertEqual(inspection.status_code,200)
                self.assertTrue(inspection.get_json()['data_split']['reserved'])
                response=self.client.post('/api/detect_upload',json={'image':image})
            self.assertEqual(response.status_code,409)
            self.assertEqual(response.get_json()['case_id'],'US_BC2021')
            self.assertNotIn('risk_level',response.get_json())
            response=self.client.post('/api/compare',json={'image_t1':image,'image_t2':image})
            self.assertEqual(response.status_code,409)

    def test_old_xinjing_raw_role_does_not_override_development_registry(self):
        from protected_inputs import input_split
        path=Path('D:/LMHLD/datasets/mine_benchmark_20261002/development/Xinjing/post_BGRN_10m.tif')
        _,meta=decode_upload(base64.b64encode(path.read_bytes()).decode(),'AUTO')
        self.assertFalse(input_split(meta)['reserved'])

    def test_reencoded_tagless_reserved_crops_rejected_before_model_loading(self):
        from validation.mine_data_protocol import registry
        from rasterio.windows import Window
        value=registry()
        for case_id in ('CL0002','KG0001','US_BC2021','TR_COP2024'):
            source=value['cases'][case_id]['protected_geographic_extent_source']
            with rasterio.open(source) as f:
                win=Window(f.width//2,f.height//2,64,64);raw=f.read(window=win);profile=f.profile.copy()
                profile.update(width=64,height=64,transform=f.window_transform(win),compress='lzw')
                with MemoryFile() as memory:
                    with memory.open(**profile) as out:
                        out.write(raw);out.descriptions=('B02','B03','B04','B08')
                    image=base64.b64encode(memory.read()).decode()
            with patch.object(service,'_load_upload_detector',side_effect=AssertionError('isolated crop must not load a model')) as loader:
                inspection=self.client.post('/api/inspect_upload',json={'image':image})
                self.assertEqual(inspection.status_code,200)
                self.assertTrue(inspection.get_json()['data_split']['reserved'])
                for endpoint,body in [('/api/detect_upload',{'image':image}),('/api/compare',{'image_t1':image,'image_t2':image})]:
                    response=self.client.post(endpoint,json=body)
                    self.assertEqual(response.status_code,409,(case_id,endpoint,response.get_json()))
                    self.assertEqual(response.get_json()['inference_status'],'reserved_test_protected')
                    self.assertNotIn('risk_level',response.get_json())
                loader.assert_not_called()

    def test_numerical_exception_is_an_unavailable_result_not_a_user_error(self):
        with patch.object(self.stub,'detect_image',side_effect=ModelNumericalError('非有限logits')):
            r=self.client.post('/api/detect_upload',json={'image':self.b64})
        self.assertEqual(r.status_code,503)
        self.assertEqual(r.get_json()['inference_status'],'failed_numerics')
        self.assertNotIn('stats',r.get_json())


if __name__=='__main__': unittest.main()
