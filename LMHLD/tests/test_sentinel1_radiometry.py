"""Physical calibration fixtures plus original source preservation checks."""
from pathlib import Path
import sys,tempfile,unittest,json
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,rasterio
from sentinel1_radiometry import calibrate_grd_sigma0
from model_registry import file_sha256


class RadiometryTests(unittest.TestCase):
    def files(self,folder,corrected=False,modern=True):
        paths=[folder/'calibration.xml',folder/'noise.xml',folder/'product.xml']
        cal='<calibration><calibrationVectorList>'
        for line in [0,4]:cal+=f'<calibrationVector><line>{line}</line><pixel>0 4</pixel><sigmaNought>2 2</sigmaNought></calibrationVector>'
        cal+='</calibrationVectorList></calibration>'
        if modern:
            noise='<noise><noiseRangeVectorList>'+''.join(f'<noiseRangeVector><line>{line}</line><pixel>0 4</pixel><noiseRangeLut>1 1</noiseRangeLut></noiseRangeVector>' for line in [0,4])+'</noiseRangeVectorList><noiseAzimuthVectorList><noiseAzimuthVector><firstAzimuthLine>0</firstAzimuthLine><lastAzimuthLine>4</lastAzimuthLine><firstRangeSample>0</firstRangeSample><lastRangeSample>4</lastRangeSample><line>0 4</line><noiseAzimuthLut>2 4</noiseAzimuthLut></noiseAzimuthVector></noiseAzimuthVectorList></noise>'
        else:
            noise='<noise><noiseVectorList>'+''.join(f'<noiseVector><line>{line}</line><pixel>0 4</pixel><noiseLut>1 1</noiseLut></noiseVector>' for line in [0,4])+'</noiseVectorList></noise>'
        product=f'<product><numberOfLines>5</numberOfLines><numberOfSamples>5</numberOfSamples><thermalNoiseCorrectionPerformed>{str(corrected).lower()}</thermalNoiseCorrectionPerformed></product>'
        for p,value in zip(paths,[cal,noise,product]):p.write_text(value,encoding='utf-8')
        return paths

    def test_original_DN_is_squared_and_range_azimuth_noise_removed(self):
        with tempfile.TemporaryDirectory(prefix='radar_cal_') as name:
            folder=Path(name);self.assertEqual(folder.drive.upper(),'D:');paths=self.files(folder)
            raw=np.full((2,2),4,np.uint16);sigma,db,valid,receipt=calibrate_grd_sigma0(raw,[1,1,2,2],*paths)
            np.testing.assert_allclose(sigma,[[3.375,3.375],[3.25,3.25]])
            np.testing.assert_allclose(db,10*np.log10(sigma),rtol=1e-6)
            self.assertTrue(valid.all());self.assertFalse(receipt['geometric_or_radiometric_terrain_correction_performed'])

    def test_already_denoised_source_is_not_subtracted_twice(self):
        with tempfile.TemporaryDirectory(prefix='radar_done_') as name:
            paths=self.files(Path(name),corrected=True)
            sigma,_,_,receipt=calibrate_grd_sigma0(np.full((2,2),4,np.uint16),[1,1,2,2],*paths)
            np.testing.assert_array_equal(sigma,np.full((2,2),4))
            self.assertEqual(receipt['thermal_noise'],'source_already_corrected_not_subtracted_again')

    def test_legacy_2017_noise_lut_supported_without_fake_azimuth_factors(self):
        with tempfile.TemporaryDirectory(prefix='radar_old_') as name:
            paths=self.files(Path(name),modern=False)
            sigma,_,_,_=calibrate_grd_sigma0(np.full((2,2),4,np.uint16),[1,1,2,2],*paths)
            np.testing.assert_array_equal(sigma,np.full((2,2),3.75))

    def test_below_noise_and_absent_samples_remain_unknown(self):
        with tempfile.TemporaryDirectory(prefix='radar_unknown_') as name:
            paths=self.files(Path(name))
            sigma,db,valid,_=calibrate_grd_sigma0(np.array([[0,1],[4,4]],np.uint16),[1,1,2,2],*paths)
            self.assertFalse(valid[0].any());self.assertTrue(np.all(sigma[0]==0));self.assertTrue(np.all(db[0]==-9999))

    def test_unverified_noise_state_or_original_window_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='radar_bad_') as name:
            paths=self.files(Path(name));paths[2].write_text('<product><numberOfLines>5</numberOfLines><numberOfSamples>5</numberOfSamples></product>',encoding='utf-8')
            with self.assertRaises(ValueError):calibrate_grd_sigma0(np.ones((2,2)),[1,1,2,2],*paths)
            paths=self.files(Path(name))
            with self.assertRaises(ValueError):calibrate_grd_sigma0(np.ones((2,2)),[4,4,2,2],*paths)

class RealRadarContractTests(unittest.TestCase):
    def test_all_real_radar_originals_remain_frozen_and_not_model_admitted(self):
        data=ROOT/'datasets/sar_development_pairs_20261005'
        for case in ['Amyntaio','LasCruces','Hpakant']:
            raw=json.loads((data/case/'raw_acquisition.json').read_text(encoding='utf-8'))
            self.assertFalse(raw['model_input_admitted']);self.assertEqual(len(raw['epochs']),2)
            for entry in raw['epochs'].values():
                for name,digest in entry['frozen_assets'].items():self.assertEqual(file_sha256(data/case/name),digest)
                for details in entry['windows'].values():
                    self.assertEqual(details['source_resolution_range_m'],20);self.assertEqual(details['source_resolution_azimuth_m'],22)
            calibration=json.loads((data/case/'radiometry_v1/calibration.json').read_text(encoding='utf-8'))
            self.assertFalse(calibration['model_input_admitted']);self.assertTrue(calibration['original_DN_images_unmodified'])

    def test_missing_exact_event_radar_is_not_zero_filled_or_claimed_available(self):
        data=ROOT/'datasets/sar_development_pairs_20261005';gap=json.loads((data/'CN_XJ2023/source_chronology_gap.json').read_text(encoding='utf-8'))
        self.assertTrue(gap['primary_manuscript_reports_SAR_gap_three_months_before_failure'])
        self.assertFalse(gap['model_input_admitted']);self.assertFalse((data/'CN_XJ2023/raw_acquisition.json').exists())

if __name__=='__main__':unittest.main(verbosity=2)
