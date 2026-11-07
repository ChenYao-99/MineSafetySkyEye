import json,sys,unittest
from pathlib import Path
ROOT=Path('D:/LMHLD');sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'cache/research_dependencies'))
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import numpy as np,rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from pixel_reference_contract import load_partial_reference,observed_core_recall
from protected_inputs import input_split
from model_registry import file_sha256
from validation.mine_data_protocol import registry,verify_assets,assert_development_cases,DATA
GAMS=ROOT/'datasets/arid_accident_operation_20261004/ZA_GAMS2020S/pixel_reference_v1_wang2026'
COP=DATA/'reserved_candidates/TR_COP2024/pixel_reference_v1_field2025'
JAGS=ROOT/'datasets/jagersfontein_development_20261005/ZA_JAGS2022/pixel_reference_v1_official_partial'
class PixelReferenceContractTests(unittest.TestCase):
    def test_real_frozen_reference_data_and_no_false_background(self):
        for folder,count in ((GAMS,38),(COP,2978)):
            d=load_partial_reference(folder);self.assertEqual(int((d['labels']==1).sum()),count)
            self.assertEqual(set(np.unique(d['labels'])),{1,255});self.assertFalse(d['reference']['source_original_survey_GIS'])
            self.assertFalse(d['reference']['accepted_for_full_accident_IoU_or_area_accuracy'])
    def test_geojson_core_reprojects_to_same_native_positive_support(self):
        for folder in (GAMS,COP):
            d=load_partial_reference(folder);geo=json.loads((folder/'published_outline_and_conservative_core_wgs84.geojson').read_text(encoding='utf-8'))
            core=next(f['geometry'] for f in geo['features'] if f['properties']['kind']=='conservative_positive_core')
            crs,grid,shape=d['grid'];native=transform_geom('EPSG:4326',crs,core)
            support=rasterize([(native,1)],out_shape=shape,transform=grid,dtype='uint8')>0
            self.assertTrue(support[d['labels']==1].all());self.assertGreaterEqual(int(support.sum()),int((d['labels']==1).sum()))
    def test_reserved_reference_cannot_enter_training_or_early_test(self):
        with self.assertRaises(ValueError):load_partial_reference(COP,use='development_training')
        with self.assertRaises(ValueError):assert_development_cases(['TR_COP2024'])
        with self.assertRaises(ValueError):load_partial_reference(COP,use='known_core_recall')
        self.assertFalse((DATA/'final_evaluation_once.json').exists())
    def test_partial_references_reject_complete_IoU(self):
        for folder in (GAMS,COP):
            with self.assertRaises(ValueError):load_partial_reference(folder,use='full_accident_iou')
    def test_unknown_prediction_values_do_not_change_known_core_recall(self):
        d=load_partial_reference(GAMS);a=np.zeros(d['labels'].shape,np.float32);b=a.copy();b[d['labels']==255]=1
        self.assertEqual(observed_core_recall(a,GAMS),observed_core_recall(b,GAMS))
        self.assertEqual(observed_core_recall(a,GAMS)['recall'],0.)
    def test_old_presence_labels_stay_all_unknown(self):
        old=GAMS.parent/'presence_reference_v1/pixel_labels_unknown.tif'
        with rasterio.open(old) as f:self.assertEqual(set(np.unique(f.read(1))),{255})
    def test_reserved_raw_and_known_preview_hashes_are_protected(self):
        paths=[COP.parent/'pre_BGRN_10m.tif',COP.parent/'post_BGRN_10m.tif',COP.parent/'pre_original_preview.png',COP/'original_pair_partial_reference_review.png',COP/'labels_partial.tif']
        for path in paths:
            split=input_split({'input_sha256':file_sha256(path)});self.assertTrue(split['reserved']);self.assertEqual(split['case_id'],'TR_COP2024')
    def test_original_assets_and_reserved_prediction_counters_unchanged(self):
        verify_assets();v=registry()
        for eid in ('CL0002','KG0001','US_BC2021','TR_COP2024'):self.assertEqual(v['cases'][eid]['model_predictions_run'],0)
        for eid in ('ZA_GAMS2020S','ZA_GAMS2023W'):
            case=v['cases'][eid]
            self.assertEqual(case.get('mine_group') or case.get('mine_id'),'ZA_GAMSBERG')
    def test_failed_direct_post_fit_stays_failed_and_chain_is_explicit(self):
        d=load_partial_reference(COP);c=d['reference']['coordinate_chain']
        self.assertTrue(c['source_to_original_pre_registration_gate_pass']);self.assertTrue(c['original_pre_post_common_grid_gate_pass'])
        self.assertFalse(c['source_post_direct_registration_gate_pass']);self.assertTrue(c['failed_direct_post_fit_not_used_for_labels'])
        self.assertFalse(c['images_or_labels_shifted_or_warped'])
    def test_official_new_mine_core_is_trainable_but_unknown_stays_unknown(self):
        d=load_partial_reference(JAGS,'development_training')
        self.assertEqual(d['case_id'],'ZA_JAGS2022');self.assertEqual(d['role'],'development')
        self.assertEqual(int((d['labels']==1).sum()),16681)
        self.assertEqual(int((d['labels']==255).sum()),386508)
        self.assertEqual(set(np.unique(d['labels'])),{1,255})
        with self.assertRaises(ValueError):load_partial_reference(JAGS,'full_accident_iou')
        with rasterio.open(JAGS/'paired_positive_support.tif') as f:paired=f.read(1)>0
        self.assertEqual(int(paired.sum()),15974);self.assertFalse((paired & (d['labels']!=1)).any())
if __name__=='__main__':unittest.main(verbosity=2)
