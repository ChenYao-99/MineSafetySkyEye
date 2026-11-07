"""A corrupt backup must not be published or overwrite existing records."""
import sqlite3,sys,unittest
from pathlib import Path
from contextlib import closing
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import backend as service
from archive_test_support import isolate_archive
from test_case_archive import fixture
from archive_recovery import restore_archive
from case_archive import CaseArchive,ArchiveError


class ArchiveRecoveryTests(unittest.TestCase):
    def setUp(self):
        isolate_archive(self,service)
        self.case=service.case_archive.create_case('内部验收','恢复保护')
        self.report=service.cache_report(fixture(),self.case['case_id'])
        service.case_archive.submit_review(self.report['report_id'],{'expected_revision':0,'target_key':'report',
            'status':'insufficient_evidence','reviewer':'内部测试','evidence':'功能测试','note':'非专业用户意见'})
        self.backup=service.case_archive.backup()
        self.destination=service.case_archive.path.parent/'restored.sqlite3'

    def test_valid_backup_preserves_facts_hashes_and_review_history(self):
        original=self.backup.read_bytes();receipt=restore_archive(self.backup,self.destination)
        restored=CaseArchive(self.destination).get_report(self.report['report_id'])
        self.assertEqual(receipt['reports'],1);self.assertEqual(receipt['reviews'],1)
        self.assertEqual(restored['objective_facts'],self.report['objective_facts'])
        self.assertEqual(restored['archive']['snapshot_sha256'],self.report['archive']['snapshot_sha256'])
        self.assertEqual(restored['archive']['review_revision'],1)
        self.assertEqual(self.backup.read_bytes(),original)

    def test_existing_target_and_sqlite_sidecars_are_never_overwritten(self):
        self.destination.write_bytes(b'OWNED_EXISTING_FILE')
        with self.assertRaises(ArchiveError):restore_archive(self.backup,self.destination)
        self.assertEqual(self.destination.read_bytes(),b'OWNED_EXISTING_FILE')
        self.destination.unlink();sidecar=Path(str(self.destination)+'-wal');sidecar.write_bytes(b'EXISTING_LOG')
        with self.assertRaises(ArchiveError):restore_archive(self.backup,self.destination)
        self.assertEqual(sidecar.read_bytes(),b'EXISTING_LOG');self.assertFalse(self.destination.exists())

    def test_corrupt_snapshot_and_missing_review_version_are_rejected_without_partial_publish(self):
        for sql in ["UPDATE reports SET payload_sha256='broken'","DELETE FROM reviews"]:
            backup=service.case_archive.backup()
            with closing(sqlite3.connect(backup)) as db:db.execute(sql);db.commit()
            with self.assertRaises(ArchiveError):restore_archive(backup,self.destination)
            self.assertFalse(self.destination.exists())
            self.assertEqual(list(self.destination.parent.glob('.archive_restore_*')),[])


if __name__=='__main__':unittest.main()
