from pathlib import Path
import copy,json,sys,tempfile,unittest
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,rasterio
from mine_work_context import annotate_known_work,REGISTRY

class KnownWorkContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=ROOT/'experiments/government_work_runtime_validation_20261005'
        cls.receipt=json.loads((folder/'default_mine_API_on_Amrun_yearly_work.json').read_text(encoding='utf-8'))
        cls.detection=cls.receipt['actual_detector_result']
        probability=np.load(folder/'default_mine_Amrun_yearly_probability.npy')
        with rasterio.open(folder/'post_original_BGRN_with_original_SCL.tif') as src:
            cls.valid=np.all(src.read_masks(indexes=[1,2,3,4])>0,axis=0)&~np.isin(src.read(5),[0,1,3,8,9,10,11])
        cls.mask=(probability>=cls.detection['threshold'])&cls.valid

    def test_actual_saved_mask_is_preserved_and_context_remains_unconfirmed(self):
        mask=self.mask.copy();stats=copy.deepcopy(self.detection['stats']);original_stats=copy.deepcopy(stats)
        result=annotate_known_work(mask,self.detection['band_meta'],self.valid,stats)
        self.assertEqual(result['status'],'matched_public_expected_work_reference')
        np.testing.assert_array_equal(mask,self.mask)
        for key,value in original_stats.items():
            if key!='regions':self.assertEqual(stats[key],value)
        self.assertFalse(result['geotechnical_safety_or_accident_absence_verified'])
        self.assertFalse(result['exact_daily_change_cause_verified'])
        self.assertGreater(result['predicted_pixels_overlapping_work_reference'],0)
        self.assertTrue(result['raw_prediction_pixels_weights_and_screening_level_unchanged'])

    def test_unknown_input_or_date_mismatch_cannot_inherit_registered_cause(self):
        meta=copy.deepcopy(self.detection['band_meta']);meta['input_sha256']='f'*64
        self.assertIsNone(annotate_known_work(self.mask,meta,self.valid,{}))
        meta['input_sha256']=self.detection['band_meta']['input_sha256'];meta['acquisition_datetime']='2026-01-01T00:00:00Z'
        result=annotate_known_work(self.mask,meta,self.valid,{})
        self.assertEqual(result['status'],'source_reference_unavailable')

    def test_ROI_and_tampered_reference_are_not_silently_accepted(self):
        valid=self.valid.copy();valid[:800]=False;mask=self.mask&valid
        result=annotate_known_work(mask,self.detection['band_meta'],valid,{})
        self.assertLessEqual(result['predicted_pixels_overlapping_work_reference'],int(mask.sum()))
        self.assertLess(result['work_reference_pixels_in_scope'],56449)
        registry=json.loads(REGISTRY.read_text(encoding='utf-8'))
        registry['entries'][self.detection['band_meta']['input_sha256']]['reference_sha256']='0'*64
        with tempfile.NamedTemporaryFile(mode='w',suffix='.json',dir=ROOT/'cache',encoding='utf-8',delete=False) as tmp:
            json.dump(registry,tmp);path=Path(tmp.name)
        try:result=annotate_known_work(self.mask,self.detection['band_meta'],self.valid,{},path)
        finally:
            if not path.resolve().is_relative_to((ROOT/'cache').resolve()):raise ValueError('Temporary path escaped owned cache')
            path.unlink()
        self.assertEqual(result['status'],'source_reference_unavailable')

if __name__=='__main__':unittest.main(verbosity=2)
