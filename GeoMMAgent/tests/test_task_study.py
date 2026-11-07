"""Task timing excludes automation from human evidence; nonce and reopen checks matter."""
from pathlib import Path
import copy,json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from task_study import TaskStudy
from case_archive import ArchiveError


class TaskStudyTests(unittest.TestCase):
    def setUp(self):
        root=Path('D:/GeoMMAgent/cache/task_study_unit');root.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.temp.cleanup)
        self.tick=0
        self.store=TaskStudy('D:/GeoMMAgent/cache/high_value_sprint_20261008/task_bank.json',self.temp.name,clock=lambda:self.tick)

    def test_real_self_report_requires_confirmation_and_cannot_skip_handoff(self):
        with self.assertRaises(ArchiveError):self.store.create({'participant_kind':'human_self_reported','role':'同学'})
        token=self.store.create({'participant_kind':'automated_test','role':'自动化功能验收'})['token']
        current=self.store.current(token)
        with self.assertRaises(ArchiveError):self.store.confirm(token,{'nonce':current['nonce'],'saved_record_checked':True})
        self.assertEqual(self.store.summary(self.store.read(token))['human_self_reported_observations'],0)

    def test_twelve_scripted_trials_preserve_timing_source_and_no_human_claim(self):
        token=self.store.create({'participant_kind':'automated_test','role':'自动化功能验收'})['token']
        for _ in range(12):
            current=self.store.current(token);self.tick+=3
            payload={'nonce':current['nonce'],'fields':current['evidence'],'note':'内部功能验收，不是专业意见','box_px':[4,4,20,20]}
            saved=self.store.submit(token,payload)
            self.assertFalse(saved['saved_record']['annotation']['verified_ground_truth'])
            with self.assertRaises(ArchiveError):self.store.submit(token,payload)
            self.tick+=4;self.store.confirm(token,{'nonce':current['nonce'],'saved_record_checked':True})
        summary=self.store.current(token)['summary']
        self.assertEqual(summary['status'],'complete');self.assertEqual(summary['human_self_reported_observations'],0)
        for arm in summary['arms'].values():
            self.assertEqual(arm['tasks_completed'],6);self.assertEqual(arm['median_total_task_seconds'],7)
            self.assertEqual(arm['incorrect_or_missing_fields'],0)

    def test_wrong_units_and_tampered_saved_handoff_are_not_success(self):
        token=self.store.create({'participant_kind':'automated_test','role':'自动化功能验收'})['token']
        current=self.store.current(token);fields=copy.deepcopy(current['evidence']);fields['unit']='invalid'
        self.store.submit(token,{'nonce':current['nonce'],'fields':fields,'note':'内部测试','box_px':[4,4,20,20]})
        state=self.store.read(token);saved=state['current']['saved_record']
        if current['arm']=='worksheet':
            path=self.store.directory/(token+'_0_worksheet.json');data=json.loads(path.read_text(encoding='utf-8'));data['review_note']='changed';path.write_text(json.dumps(data),encoding='utf-8')
        else:
            with self.store.archive.connection() as db:db.execute('DELETE FROM reviews WHERE report_id=?',(saved['report_id'],))
        with self.assertRaises(ArchiveError):
            self.store.confirm(token,{'nonce':current['nonce'],'saved_record_checked':True})
        self.assertFalse(state['current']['grades']['unit'])
        self.assertEqual(self.store.read(token)['index'],0)
