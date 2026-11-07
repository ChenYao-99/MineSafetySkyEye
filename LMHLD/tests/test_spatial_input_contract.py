"""Scale replay,physical area conservation and invalid-coverage safety."""
from pathlib import Path
import sys,unittest,base64
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,cv2
import json,csv,io
from rasterio.transform import from_origin
from spatial_input_contract import SpatialInputContract
from test_geotiff_upload import tiff_data,Stub
import detector_service as service


class SpatialContracts(unittest.TestCase):
    def image(self):
        x=np.stack([np.full((32,32),v,np.uint16) for v in [30,40,50,90]])
        x[0,4:12,4:12]=200
        return x

    def test_same_ground_information_returns_same_mask_area_and_region_count(self):
        original=self.image();replica=np.repeat(np.repeat(original,4,axis=1),4,axis=2)
        client=service.app.test_client()
        with patch.object(service,'_load_upload_detector',return_value=(Stub(),'minmax',1)):
            native=client.post('/api/detect_upload',json={'image':tiff_data(original)}).get_json()
            response=client.post('/api/detect_upload',json={'image':tiff_data(replica,transform=from_origin(754950,4156920,2.5,2.5)),'analysis_grid':'reference_10m'})
        self.assertEqual(response.status_code,200);mapped=response.get_json()
        self.assertEqual(mapped['stats']['landslide_area_m2'],native['stats']['landslide_area_m2'])
        self.assertEqual(mapped['stats']['area_ratio'],native['stats']['area_ratio'])
        self.assertEqual(mapped['stats']['num_regions'],native['stats']['num_regions'])
        self.assertEqual(native['size'],[32,32]);self.assertEqual(mapped['size'],[128,128])
        self.assertEqual(mapped['band_meta']['spatial_input_contract']['analysis_shape'],[32,32])
        self.assertFalse(mapped['band_meta']['spatial_input_contract']['accuracy_on_real_high_resolution_sensors_verified'])
        exported=client.post('/api/geometry_export',json={'regions':mapped['stats']['regions'],'format':'geojson',
            'georeference':mapped['band_meta']['georeference'],'meters_per_pixel':2.5,
            'spatial_input_contract':mapped['band_meta']['spatial_input_contract']}).get_json()
        document=json.loads(base64.b64decode(exported['content_b64']))
        self.assertEqual(document['spatial_input_contract']['reference_meters_per_pixel'],10)
        self.assertFalse(document['spatial_input_contract']['ground_boundary_precision_verified'])
        exported_csv=client.post('/api/geometry_export',json={'regions':mapped['stats']['regions'],'format':'csv',
            'meters_per_pixel':2.5,'spatial_input_contract':mapped['band_meta']['spatial_input_contract']}).get_json()
        rows=list(csv.DictReader(io.StringIO(base64.b64decode(exported_csv['content_b64']).decode())))
        self.assertTrue(rows)
        self.assertEqual(rows[0]['mode'],'reference_10m')
        self.assertEqual(float(rows[0]['source_meters_per_pixel']),2.5)
        self.assertEqual(json.loads(rows[0]['effective_pixel_size_m']),[10,10])
        self.assertEqual(rows[0]['ground_boundary_precision_verified'],'False')
        self.assertIn('不增加边界细节',exported_csv['note'])

    def test_mixed_nodata_cell_cannot_be_invented_as_fully_observed(self):
        image=np.ones((128,128,4),np.float32)*100;image[30:60,:,0]=200
        valid=np.ones((128,128),bool);valid[0,0]=False
        contract=SpatialInputContract(image,valid,2.5,'reference_10m')
        self.assertFalse(contract.valid[0,0]);self.assertFalse(contract.native_support[:4,:4].any())
        self.assertEqual(contract.record['inference_excluded_native_pixels'],15)

    def test_display_uses_reported_region_filter_without_claiming_raw_noise_as_an_instance(self):
        original=self.image();original[0,25,25]=200
        replica=np.repeat(np.repeat(original,4,axis=1),4,axis=2)
        client=service.app.test_client()
        for image,mpp,mode in [(original,10,'native'),(replica,2.5,'reference_10m')]:
            detector=Stub();seen=[]
            def overlay(rgb,mask):
                seen.append(mask.copy());return rgb
            with patch.object(service,'_load_upload_detector',return_value=(detector,'minmax',1)),patch.object(detector,'overlay',side_effect=overlay):
                response=client.post('/api/detect_upload',json={'image':tiff_data(image,transform=from_origin(754950,4156920,mpp,mpp)),'analysis_grid':mode})
            self.assertEqual(response.status_code,200)
            result=response.get_json();contract=result['mask_display_contract']
            self.assertEqual(int(seen[0].sum()),result['stats']['landslide_pixels'])
            self.assertEqual(result['stats']['num_regions'],1)
            self.assertEqual(contract['small_component_excluded_pixels'],1 if mode=='native' else 16)
            self.assertFalse(contract['classification_threshold_changed'])

    def test_api_distinguishes_actual_nodata_from_inference_cell_exclusions(self):
        replica=np.repeat(np.repeat(self.image(),4,axis=1),4,axis=2)
        replica[:,0,0]=0
        with patch.object(service,'_load_upload_detector',return_value=(Stub(),'minmax',1)):
            response=service.app.test_client().post('/api/detect_upload',json={
                'image':tiff_data(replica,transform=from_origin(754950,4156920,2.5,2.5)),
                'analysis_grid':'reference_10m'})
        self.assertEqual(response.status_code,200)
        stats=response.get_json()['stats']
        self.assertEqual(stats['source_nodata_pixels'],1)
        self.assertEqual(stats['inference_excluded_pixels'],15)
        self.assertEqual(stats['nodata_pixels'],16)
        self.assertEqual(stats['valid_pixels'],128*128-16)

    def test_undefined_scale_coarse_data_small_context_and_other_model_refused_before_load(self):
        client=service.app.test_client();x=self.image();ok,png=cv2.imencode('.png',x.transpose(1,2,0));self.assertTrue(ok)
        payloads=[{'image':base64.b64encode(png).decode()},
            {'image':tiff_data(x,transform=from_origin(754950,4156920,20,20))},
            {'image':tiff_data(x,transform=from_origin(754950,4156920,.5,.5))},
            {'image':tiff_data(x),'model_tag':'base'}]
        with patch.object(service,'_load_upload_detector',side_effect=AssertionError('must not load')):
            for payload in payloads:
                response=client.post('/api/detect_upload',json={**payload,'analysis_grid':'reference_10m'})
                self.assertEqual(response.status_code,400,response.get_json())

    def test_reserved_metadata_is_checked_before_any_spatial_route(self):
        client=service.app.test_client()
        with patch.object(service,'input_split',return_value={'reserved':True,'case_id':'protected'}),patch.object(service,'SpatialInputContract',side_effect=AssertionError('must not preprocess')):
            response=client.post('/api/detect_upload',json={'image':tiff_data(self.image()),'analysis_grid':'reference_10m'})
        self.assertEqual(response.status_code,409)


if __name__=='__main__':unittest.main()
