"""Regress actual competition-report errors without spending API calls."""
from pathlib import Path
from unittest.mock import patch
import copy,json,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,'D:/LMHLD')
from evidence_reporting import bounded_input
from report_grounding import check_known_input_claims,check_detection_numbers,report_input_context,fallback_report
from report_workflow import ReportCoordinator
from geo_tools import compare_periods,format_quant_block


class CompetitionReportSemantics(unittest.TestCase):
    def setUp(self):
        self.roi={'stats':{'scope':'roi','area_ratio':.0936,'num_regions':31},
            'roi':{'box_px':[77,86,308,347]},
            'band_meta':{'acquisition_datetime':'2019-01-27T11:13:29Z','nir_source':'measured',
                         'spectral_indices_available':True,'radiometry_status':'verified_local_product_XML',
                         'georeference':{'independent_ground_accuracy_verified':False}}}
        self.pair={'kind':'compare','stats':{'t1_ratio':.0884,'t2_ratio':.0889,'net_ratio':.0005,
                   'n_added_regions':78,'n_expanded_regions':49,'n_removed_regions':58}}

    def test_pixel_ROI_and_canonical_percentage_points_are_actually_supplied(self):
        value=bounded_input(self.roi)
        self.assertTrue(value['roi_selected'])
        self.assertEqual(value['roi_box_px'],[77,86,308,347])
        self.assertEqual(value['objective_facts']['area_percent'],9.36)
        self.assertEqual(bounded_input(self.pair)['objective_facts']['change']['net_percentage_points'],.05)
        context=report_input_context(self.roi)
        self.assertIn('"roi_selected":true',context)
        self.assertNotIn('EPSG:',context)

    def test_known_selection_is_distinct_from_engineering_suitability(self):
        for text in ['ROI范围未明确圈定。','当前ROI是否已圈定且与工具scope字段一致。']:
            self.assertTrue(check_known_input_claims(text,self.roi),text)
        for text in ['ROI是否与现场工程分区一致仍需复核。','后续若ROI未明确圈定，应先选择范围。']:
            self.assertEqual(check_known_input_claims(text,self.roi),[],text)

    def test_unknown_ROI_is_not_upgraded_to_confirmed_selection(self):
        self.roi['roi']={}
        self.assertIsNone(bounded_input(self.roi)['roi_selected'])
        self.assertEqual(check_known_input_claims('ROI范围未明确圈定。',self.roi),[])
        self.roi['stats']['scope']='whole_image'
        self.assertFalse(bounded_input(self.roi)['roi_selected'])

    def test_provided_NIR_is_distinct_from_field_validation(self):
        self.assertTrue(check_known_input_claims('未提供真实NIR波段。',self.roi))
        self.assertEqual(check_known_input_claims('未提供真实NIR波段细节和独立地面验证。',self.roi),[])
        self.roi['band_meta']['nir_source']='red_proxy'
        self.assertEqual(check_known_input_claims('未提供真实NIR波段。',self.roi),[])

    def test_product_calibration_is_not_independent_field_calibration(self):
        self.assertTrue(check_known_input_claims('未说明影像大气校正与辐射定标状态。',self.roi))
        self.assertEqual(check_known_input_claims('未提供独立地面辐射定标验证。',self.roi),[])

    def test_different_rounded_values_need_a_rounding_explanation(self):
        self.assertTrue(check_detection_numbers('净变化+0.05个百分点（即+0.1个百分点，按百分点计）。',self.pair))
        self.assertEqual(check_detection_numbers('净变化+0.05个百分点（即+0.1个百分点，按保留一位小数舍入显示）。',self.pair),[])

    def test_zero_baseline_has_no_fabricated_relative_growth(self):
        comparison=compare_periods({'area_ratio':0},{'area_ratio':.01})
        self.assertIsNone(comparison['change_rate'])
        block=format_quant_block(compare=comparison)
        self.assertIn('净变化 +1.00 个百分点',block)
        self.assertNotIn('变化率',block)
        self.assertNotIn('风险扩大',block)

    def test_small_real_change_survives_prompt_and_fallback_display(self):
        comparison=compare_periods({'area_ratio':.0884},{'area_ratio':.0889})
        block=format_quant_block(compare=comparison)
        self.assertIn('8.84% → 8.89%',block)
        self.assertIn('+0.05 个百分点',block)
        text=fallback_report(self.pair,'local acceptance')
        self.assertIn('净变化+0.05个百分点',text)
        self.assertEqual(check_detection_numbers(text,self.pair),[])

    def test_recorded_audit_errors_are_rejected_before_the_next_role(self):
        receipt=json.loads((ROOT/'maintenance/phase5_20261014/cloud_acceptance.json').read_text('utf-8'))
        for task,code in [('M02','audit_numeric_claim_inconsistent'),('M03','audit_provided_condition_called_unknown')]:
            row=next(x for x in receipt['rows'] if x['task_id']==task)
            det=json.loads((ROOT/('maintenance/phase4_20261013/cloud_inputs/'+task+'_detector.json')).read_text('utf-8'))
            coord=ReportCoordinator.__new__(ReportCoordinator)
            coord._evidence_context=row['response']['quality']['evidence_context']
            coord.facts=det
            self.assertIn(code,coord.validate_step('EvidenceAuditAgent',row['response']['agents'][0]['output']))


if __name__=='__main__':unittest.main(verbosity=2)
