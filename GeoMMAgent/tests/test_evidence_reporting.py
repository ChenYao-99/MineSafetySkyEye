import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path('D:/GeoMMAgent')))
from evidence_reporting import objective_facts,bounded_input,fixed_evidence_report,score_objective_report

class ObjectiveReportingTests(unittest.TestCase):
    def setUp(self):
        self.d=json.loads(Path('D:/GeoMMAgent/cache/competition_readiness_20261005/real_detection.json').read_text(encoding='utf-8'))
    def test_real_measurement_fields_and_known_unknown_conditions(self):
        f=objective_facts(self.d);self.assertEqual(f['area_percent'],8.63)
        self.assertEqual(f['largest_area_px'],1481);self.assertEqual(f['largest_area_m2'],148100)
        self.assertFalse(f['weather_quality_provided']);self.assertFalse(f['site_stability_confirmed'])
        self.assertTrue(f['physical_spectral_indices_available'])
    def test_fixed_fields_correct_model_units_without_asserting_accident(self):
        wanted=objective_facts(self.d);draft={'facts':copy.deepcopy(wanted),'summary':'边坡处于安全状态','citations':['FAKE-001']}
        draft['facts']['largest_area_m2']=1481;draft['facts']['largest_perimeter_px']=wanted['largest_major_axis_px']
        wrong=score_objective_report(draft,wanted,[]);self.assertEqual(wrong['wrong_fields'],2)
        fixed=fixed_evidence_report(draft,self.d,[]);self.assertTrue(score_objective_report(fixed,wanted,[])['all_objective_fields_correct'])
        self.assertEqual(fixed['citations'],[]);self.assertNotIn('处于安全',fixed['summary'])
    def test_unavailable_fraction_and_pair_percentage_points(self):
        d=copy.deepcopy(self.d);d['stats']['area_ratio_available']=False
        self.assertIsNone(objective_facts(d)['area_percent'])
        d['stats']={'t1_ratio':.3,'t2_ratio':.25,'net_ratio':-.05,'n_added_regions':2,'n_expanded_regions':4,'n_removed_regions':7}
        d['band_meta']={'t1':{},'t2':{}};f=objective_facts(d)
        self.assertEqual(f['change']['net_percentage_points'],-5);self.assertEqual(f['change']['expanded_regions'],4)
        self.assertIsNone(f['reported_regions']);self.assertFalse(f['ground_displacement_measured'])

    def test_outbound_minimum_fields_remove_originals_coordinates_and_private_metadata(self):
        d=copy.deepcopy(self.d)
        d['band_meta']['georeference']={'crs':'EPSG:32629','transform':[10,0,123456,0,-10,654321],
                                      'independent_ground_accuracy_verified':False}
        d['band_meta']['original_path']='D:/private/original.tif'
        d['band_meta']['api_key']='PRIVATE_SENTINEL'
        d['stats']['regions'][0].update({'bbox':[1,2,3,4],'centroid':[2,3],'contour':[[1,2]]})
        outgoing=bounded_input(d);text=json.dumps(outgoing)
        for forbidden in ('original_path','api_key','PRIVATE_SENTINEL','transform','EPSG:','bbox','centroid','contour','overlay_b64'):
            self.assertNotIn(forbidden,text)
        self.assertEqual(outgoing['first_returned_region']['area_m2'],148100)
        self.assertEqual(outgoing['band_meta']['acquisition_datetime'],d['band_meta'].get('acquisition_datetime'))

    def test_omitted_unknown_fields_count_as_errors(self):
        f=objective_facts(self.d);report={'facts':copy.deepcopy(f),'summary':'待人工复核','citations':[]}
        del report['facts']['change']['added_regions']
        self.assertEqual(score_objective_report(report,f,[])['wrong_fields'],1)

    def test_correct_structured_fields_cannot_hide_wrong_summary_units_or_fake_citations(self):
        f=objective_facts(self.d)
        draft={'facts':copy.deepcopy(f),'summary':'最大实例面积1481平方米。[FAKE-001]','citations':[]}
        score=score_objective_report(draft,f,[],det=self.d)
        self.assertTrue(score['all_objective_fields_correct']);self.assertFalse(score['report_contract_passed'])
        fixed=fixed_evidence_report(draft,self.d,[])
        self.assertTrue(score_objective_report(fixed,f,[],det=self.d)['report_contract_passed'])
        self.assertNotIn('1481平方米',fixed['summary'])

if __name__=='__main__':unittest.main(verbosity=2)
