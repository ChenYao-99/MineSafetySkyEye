"""Missing model objects can be located without changing machine facts or ground truth."""
from pathlib import Path
import base64,copy,json,sys,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import backend as service
from case_archive import CaseArchive
from archive_recovery import validate_archive,restore_archive
from archive_test_support import isolate_archive
from test_case_archive import fixture


class ManualAnnotations(unittest.TestCase):
    def setUp(self):
        isolate_archive(self,service);self.client=service.app.test_client()
        value=fixture();value['detection']['size']=[20,30]
        value['detection']['stats'].update(regions=[],num_regions=0,num_regions_total=0,landslide_pixels=0,area_ratio=0)
        self.saved=service.cache_report(value);self.rid=self.saved['report_id']
        self.payload={'expected_revision':0,'box_px':[3,4,12,15],'reviewer':'内部功能测试',
            'evidence':'本次原图显示；不是专家评审','note':'此处需要补查，不能确认事故原因'}

    def post(self,payload=None,rid=None):
        return self.client.post('/api/archive/reports/'+(rid or self.rid)+'/manual_annotations',json=payload or self.payload)

    def test_empty_prediction_can_have_bound_manual_location_exported_and_restored(self):
        response=self.post();self.assertEqual(response.status_code,200);value=response.get_json()
        self.assertEqual(value['detection'],self.saved['detection'])
        self.assertEqual(value['objective_facts'],self.saved['objective_facts'])
        self.assertEqual(value['archive']['snapshot_sha256'],self.saved['archive']['snapshot_sha256'])
        record=value['review_handoff'];self.assertEqual(record['review_items'],[])
        marker=record['manual_annotations'][0]['manual_annotation']
        self.assertEqual(marker['box_px'],[3,4,12,15]);self.assertFalse(marker['verified_ground_truth'])
        self.assertFalse(marker['geometry_is_accident_boundary']);self.assertIsNone(marker['physical_failure_area_m2'])
        exported=self.client.post('/api/review_record',json={'report_id':self.rid}).get_json()
        recovered=json.loads(base64.b64decode(exported['content_b64']))
        self.assertEqual(recovered['manual_annotations'][0]['manual_annotation'],marker)
        backup=service.case_archive.backup();destination=service.case_archive.path.parent/'manual_restored.sqlite3'
        restore_archive(backup,destination);restored=CaseArchive(destination).get_report(self.rid)
        self.assertEqual(restored['review_handoff']['manual_annotations'],record['manual_annotations'])
        self.assertEqual(validate_archive(destination)['reviews'],1)

    def test_bounds_unknown_fields_and_stale_writes_do_not_change_records(self):
        for box in [[-1,0,4,5],[0,0,31,20],[0,0,1,float('nan')],[False,0,4,5],[2,2,2,4]]:
            self.assertEqual(self.post({**self.payload,'box_px':box}).status_code,400)
        self.assertEqual(self.post({**self.payload,'verified_ground_truth':True}).status_code,400)
        self.assertEqual(self.post().status_code,200)
        self.assertEqual(self.post().status_code,409)
        self.assertEqual(service.case_archive.get_report(self.rid)['archive']['review_revision'],1)

    def test_manual_location_note_can_be_revised_and_withdrawn_without_changing_geometry(self):
        value=self.post().get_json();target=value['review_handoff']['manual_annotations'][0]['target_key']
        for revision,status in [(1,'insufficient_evidence'),(2,'withdrawn')]:
            response=self.client.post('/api/archive/reports/'+self.rid+'/reviews',json={
                'expected_revision':revision,'target_key':target,'status':status,'reviewer':'内部功能测试',
                'evidence':'补记位置仍未经过独立核验','note':'保留原图关联，更新意见'})
            self.assertEqual(response.status_code,200)
        record=response.get_json()['review_handoff']
        self.assertEqual(record['manual_annotations'],[]);self.assertEqual(len(record['manual_annotation_history']),3)
        self.assertTrue(all(row['manual_annotation']['box_px']==[3,4,12,15] for row in record['manual_annotation_history']))
        self.assertEqual(validate_archive(service.case_archive.path)['reviews'],3)

    def test_pair_is_bound_to_t2_and_cross_report_target_or_tampering_is_rejected(self):
        value=fixture(True);value['detection']['size']=[20,30];value['detection']['band_meta']['t2']['input_sha256']='c'*64
        saved=service.cache_report(value);paired=self.post(rid=saved['report_id']).get_json()
        marker=paired['review_handoff']['manual_annotations'][0]['manual_annotation']
        self.assertEqual(marker['source']['coordinate_frame'],'t2_comparison_grid');self.assertEqual(marker['source']['input_sha256'],'c'*64)
        target=paired['review_handoff']['manual_annotations'][0]['target_key']
        response=self.client.post('/api/archive/reports/'+self.rid+'/reviews',json={**self.payload,
            'target_key':target,'status':'needs_followup'})
        self.assertEqual(response.status_code,400)
        with service.case_archive.connection() as db:
            event=db.execute('SELECT note FROM reviews WHERE report_id=?',(saved['report_id'],)).fetchone()[0]
            body=json.loads(event);body['annotation']['source']['input_sha256']='d'*64
            db.execute('UPDATE reviews SET note=? WHERE report_id=?',(json.dumps(body),saved['report_id']))
        self.assertEqual(self.client.get('/api/archive/reports/'+saved['report_id']).status_code,503)


if __name__=='__main__':unittest.main()
