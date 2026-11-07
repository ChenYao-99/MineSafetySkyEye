import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from corrected_gate_inputs import load_family_rows


class CorrectedGateInputs(unittest.TestCase):
    def test_real_family_uses_correction_and_keeps_native_metrics(self):
        folder=ROOT/'experiments/rgb_direct_context_20261004'
        original=json.loads((folder/'results.json').read_text(encoding='utf-8'))['models']
        rows=load_family_rows(folder);self.assertEqual(len(rows),8)
        for old,new in zip(original,rows):
            self.assertEqual(old['old_reference'],new['old_reference'])
            self.assertTrue(new['corrected_evaluation_not_algorithm_gain'])
        amy=next(r for r in rows if r['held']=='Amyntaio' and r['arm']=='RGBpair')
        self.assertGreater(amy['source_RGB']['F1'],.5)
        self.assertLess(amy['source_RGB_previous_invalid_preprocessing']['F1'],.05)
    def test_missing_replacement_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'cache/tests') as tmp:
            folder=Path(tmp);(folder/'results.json').write_text(json.dumps({'models':[{'id':'missing'}]}))
            corrected=folder/'corrected.json';corrected.write_text(json.dumps({'models':[]}))
            (folder/'source_evaluation_correction.json').write_text(json.dumps({'source_RGB_original_evaluation_valid':False,'replacement_source_regression':str(corrected)}))
            with self.assertRaisesRegex(ValueError,'Incomplete'):load_family_rows(folder)
    def test_missing_correction_sidecar_cannot_reenable_invalid_scores(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'cache/tests') as tmp:
            folder=Path(tmp)/'rgb_direct_context_20261004';folder.mkdir()
            (folder/'results.json').write_text(json.dumps({'models':[]}))
            with self.assertRaisesRegex(ValueError,'required'):load_family_rows(folder)


if __name__=='__main__':unittest.main(verbosity=2)
