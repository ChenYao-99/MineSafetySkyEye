"""Real local-browser boundaries and streamed-backup cleanup, without model calls."""
from pathlib import Path
import sys,unittest,sqlite3,hashlib
from contextlib import closing
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import backend as service
from archive_test_support import isolate_archive


class LocalAccessAndBackup(unittest.TestCase):
    def setUp(self):
        isolate_archive(self,service);self.client=service.app.test_client()

    def test_foreign_websites_and_null_file_origins_cannot_read_or_write_archive(self):
        with patch.object(service,'detector_request',side_effect=AssertionError('must not call model')):
            for origin in ['https://unrelated.example','null','http://localhost:9999']:
                for method,path in [('get','/api/cases'),('post','/api/case_archive/backup'),('post','/api/detect_upload')]:
                    response=getattr(self.client,method)(path,headers={'Origin':origin},json={} if method=='post' else None)
                    self.assertEqual(response.status_code,403)
                    self.assertNotIn('Access-Control-Allow-Origin',response.headers)
            self.assertEqual(self.client.get('/api/cases',headers={'Host':'unrelated.example'}).status_code,403)
            self.assertEqual(self.client.get('/api/cases',headers={'Sec-Fetch-Site':'cross-site'}).status_code,403)

    def test_same_local_origin_preflight_and_cli_are_supported(self):
        response=self.client.options('/api/cases',base_url='http://127.0.0.1:5100',headers={'Origin':'http://localhost:5100','Access-Control-Request-Method':'POST'})
        self.assertEqual(response.status_code,200);self.assertEqual(response.headers['Access-Control-Allow-Origin'],'http://localhost:5100')
        self.assertEqual(self.client.get('/api/cases').status_code,200)

    def test_backup_stream_completion_or_early_close_removes_only_its_temporary_file(self):
        service.case_archive.create_case('内部测试','备份清理')
        for consume in [True,False]:
            destination=service.case_archive.backup()
            with patch.object(service.case_archive,'backup',return_value=destination):
                response=self.client.post('/api/case_archive/backup',json={},buffered=False)
                self.assertEqual(response.status_code,200);self.assertTrue(destination.exists())
                if consume:self.assertTrue(response.get_data().startswith(b'SQLite format 3'))
                else:next(iter(response.response))
                response.close()
            self.assertFalse(destination.exists());self.assertTrue(service.case_archive.path.exists())

    def test_unrelated_or_incomplete_sqlite_file_is_rejected_without_schema_changes(self):
        from case_archive import CaseArchive,ArchiveError
        parent=service.case_archive.path.parent
        for version in [0,1]:
            path=parent/('unrelated_'+str(version)+'.sqlite3')
            with closing(sqlite3.connect(path)) as db:
                db.execute('CREATE TABLE unrelated(value TEXT)');db.execute('PRAGMA user_version='+str(version))
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ArchiveError):CaseArchive(path).list_cases()
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)
            with closing(sqlite3.connect(path)) as db:self.assertEqual([r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")],['unrelated'])

    def test_empty_archive_can_be_backed_up_and_low_disk_space_fails_before_allocating(self):
        from types import SimpleNamespace
        with patch('case_archive.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
            response=self.client.post('/api/case_archive/backup',json={})
        self.assertEqual(response.status_code,503);self.assertFalse(service.case_archive.path.exists())
        response=self.client.post('/api/case_archive/backup',json={},buffered=False)
        self.assertEqual(response.status_code,200);self.assertTrue(response.get_data().startswith(b'SQLite format 3'))
        response.close()


if __name__=='__main__':unittest.main()
