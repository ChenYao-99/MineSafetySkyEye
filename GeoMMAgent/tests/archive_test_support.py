"""Keep all synthetic/test analyses out of the user's persistent archive."""
from pathlib import Path
import tempfile
from unittest.mock import patch
from case_archive import CaseArchive


def isolate_archive(test,service):
    root=Path('D:/GeoMMAgent/cache/archive_test_runtime');root.mkdir(exist_ok=True)
    directory=tempfile.TemporaryDirectory(dir=root)
    test.addCleanup(directory.cleanup)
    mocked=patch.object(service,'case_archive',CaseArchive(Path(directory.name)/'cases.sqlite3'))
    mocked.start();test.addCleanup(mocked.stop)
