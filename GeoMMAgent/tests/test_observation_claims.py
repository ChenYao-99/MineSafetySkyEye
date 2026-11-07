"""Observation guards must reject explicit inventions and preserve valid caveats."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from report_grounding import (check_observation_claims, report_input_context,
                             check_detection_numbers,check_unsupported_safety_claims,build_grounding,check_structured_geometry)
from report_grounding import check_source_detail_claims


class ObservationClaims(unittest.TestCase):
    def setUp(self):
        self.facts={'case_id':'case_current_001','kind':'single','stats':{},
                    'band_meta':{'acquisition_datetime':'2020-12-31T10:05:00Z',
                                 'spectral_indices_available':False,'nir_source':'red_proxy'}}

    def test_correct_capture_does_not_invalidate_reference_publication_date(self):
        text='本期影像采集日期为2020-12-31，案例编号：case_current_001。参考资料发布于2022-05-01。'
        self.assertEqual(check_observation_claims(text,self.facts),[])
        self.assertTrue(check_observation_claims(text.replace('采集日期为2020-12-31','采集日期为2021-01-01'),self.facts))

    def test_unknown_date_and_invalid_calendar_date_stay_unconfirmed(self):
        self.facts['band_meta']['acquisition_datetime']=None
        self.assertTrue(check_observation_claims('影像采集日期为2020年12月31日。',self.facts))
        self.assertTrue(check_observation_claims('影像采集日期为2020-02-31。',self.facts))
        self.assertEqual(check_observation_claims('影像采集日期尚未提供，请核对原始元数据。',self.facts),[])

    def test_pair_dates_and_NDVI_use_the_matching_epoch(self):
        first=copy.deepcopy(self.facts['band_meta'])
        second=copy.deepcopy(first);second.update(acquisition_datetime='2021-01-05T10:05:00Z',spectral_indices_available=True)
        self.facts.update(kind='compare',band_meta={'t1':first,'t2':second})
        self.assertEqual(check_observation_claims('T1影像采集日期为2020-12-31；T2影像采集日期为2021-01-05。T2测得NDVI为0.3。',self.facts),[])
        self.assertTrue(check_observation_claims('T1测得NDVI为0.3。',self.facts))
        self.assertTrue(check_observation_claims('T2影像采集日期为2020-12-31。',self.facts))

    def test_caveats_are_not_positive_measurement_claims(self):
        for text in ('没有测得NDVI为0.72的依据。','无法预测未来48小时内发生滑坡。',
                     '不能认为现场实测位移12毫米。','本系统尚未证明提前12天预警。'):
            self.assertEqual(check_observation_claims(text,self.facts),[],text)
        for text in ('测得NDVI为0.72。','预计未来48小时内发生滑坡。',
                     '现场实测位移12毫米。','本系统可以提前12天预警。'):
            self.assertTrue(check_observation_claims(text,self.facts),text)

    def test_foreign_case_and_unknown_pixels_cannot_be_promoted(self):
        self.facts['stats']['unverified_pixels']=350
        self.assertTrue(check_observation_claims('案例编号：case_other_999。',self.facts))
        self.assertTrue(check_observation_claims('冲突像素已归为正常背景。',self.facts))
        self.assertEqual(check_observation_claims('冲突像素不能计入正常背景，保持待复核。',self.facts),[])

    def test_repair_context_retains_unknowns_without_private_metadata(self):
        self.facts['band_meta'].update(original_path='PRIVATE_PATH_SENTINEL',secret='SECRET_SENTINEL')
        context=report_input_context(self.facts)
        self.assertIn('case_current_001',context)
        self.assertIn('2020-12-31',context)
        self.assertIn('false',context)
        self.assertNotIn('PRIVATE_PATH_SENTINEL',context)
        self.assertNotIn('SECRET_SENTINEL',context)

    def test_reference_threshold_list_is_not_current_coverage(self):
        facts={'stats':{'area_ratio':.0871,'num_regions':86}}
        text='本次异常面积占比8.7%。以预测掩膜面积占比1%、5%、15%为蓝黄橙红分级阈值。'
        self.assertEqual(check_detection_numbers(text,facts),[])
        self.assertTrue(check_detection_numbers('本次异常面积占比1%，分级阈值5%。',facts))

    def test_adjacent_change_counts_are_not_cross_bound(self):
        facts={'kind':'compare','stats':{'n_added_regions':78,'n_expanded_regions':49,'n_removed_regions':58}}
        self.assertEqual(check_detection_numbers('78处新增与49处扩大，消退58处。',facts),[])
        self.assertTrue(check_detection_numbers('新增79处，扩大49处。',facts))

    def test_stability_study_and_safety_monitoring_are_not_safety_assertions(self):
        self.assertEqual(check_unsupported_safety_claims('建议开展边坡安全监测和边坡稳定性研究。',self.facts),[])
        self.assertTrue(check_unsupported_safety_claims('边坡处于稳定状态，矿区安全。',self.facts))

    def test_geometry_prompt_uses_original_id_after_area_sorting(self):
        sys.path.insert(0,'D:/LMHLD')
        from geo_geometry import format_geometry_block
        regions=[{'id':32,'bbox':[0,0,59,31],'area_px':1481,'perimeter_px':172.08,
                  'major_axis_px':61.5,'circularity':.6285}]
        block=format_geometry_block(regions,max_chars=500)
        self.assertIn('实例#32',block)
        self.assertEqual(check_structured_geometry(block,build_grounding(regions))[1],[])

    def test_generic_ROI_reference_does_not_override_whole_image_scope(self):
        self.facts['stats']['scope']='whole_image'
        self.assertTrue(check_observation_claims('所有面积占比均限定于当前ROI范围内。',self.facts))
        self.assertEqual(check_observation_claims('后续若选择ROI范围，必须统一口径。',self.facts),[])

    def test_range_in_reference_rule_is_not_current_coverage(self):
        self.assertEqual(check_detection_numbers('以预测掩膜面积占比达5%–15%划为橙色。',{'stats':{'area_ratio':.0871}}),[])

    def test_unprovided_chapter_and_automatic_downgrade_are_not_invented(self):
        question='[REG-002] GB/T 37573-2019，仅核对标准名称，未提供全文。'
        self.assertTrue(check_source_detail_claims('该标准第6章规定年度稳定性评价指标。',question))
        self.assertTrue(check_source_detail_claims('依据GB/T 99999-2026。',question))
        self.assertEqual(check_source_detail_claims('GB/T 37573-2019全文尚待核对。',question),[])
        self.assertTrue(check_observation_claims('若20日内未获取有效位移数据或现场确认，则自动降级为常规巡检项。',self.facts))
        self.assertTrue(check_observation_claims('原型筛查分级仅为模型置信度提示。',self.facts))
        self.assertEqual(check_observation_claims('不能因为未获取现场数据就自动降级。',self.facts),[])

    def test_markdown_case_label_and_capture_epoch_are_checked(self):
        self.assertTrue(check_observation_claims('**案例编号**：case_other_999。',self.facts))
        self.assertTrue(check_observation_claims('采集时相：2021-01-01。',self.facts))

    def test_disabled_self_evaluation_does_not_make_hidden_cloud_calls(self):
        import backend as service
        from unittest.mock import patch
        with patch.object(service,'_cfg',{'agents':{'SelfEvaluationAgent':{'enabled':False}}}), \
             patch('toolkit.evaluation_metrics.run_self_evaluation_metrics',side_effect=AssertionError('unexpected evaluation call')):
            self.assertIsNone(service.run_self_eval('question',[],'answer',None))


if __name__=='__main__':unittest.main(verbosity=2)
