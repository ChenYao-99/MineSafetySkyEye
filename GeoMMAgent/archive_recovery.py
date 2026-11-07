"""Validate a consistent SQLite snapshot before publishing to a new D: path."""
from contextlib import closing
from pathlib import Path
import os,sqlite3,tempfile
from case_archive import ArchiveError,CaseArchive,REVIEW_LABELS


def validate_archive(path):
    archive=CaseArchive(path)
    with archive.connection() as db:
        db.execute('BEGIN')
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok' or db.execute('PRAGMA foreign_key_check').fetchall():
            raise ArchiveError('备份数据库结构或引用完整性校验失败',503)
        reports=[dict(row) for row in db.execute('SELECT report_id,review_revision FROM reports')]
        cases=db.execute('SELECT COUNT(*) FROM cases').fetchone()[0]
        reviews=db.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]
        if any(row[0] not in REVIEW_LABELS for row in db.execute('SELECT DISTINCT status FROM reviews')):
            raise ArchiveError('备份包含无法识别的复核状态',503)
        for row in reports:
            chain=db.execute('SELECT COUNT(*),MIN(revision),MAX(revision) FROM reviews WHERE report_id=?',(row['report_id'],)).fetchone()
            expected=row['review_revision']
            if expected<0 or chain[0]!=expected or (expected and (chain[1]!=1 or chain[2]!=expected)):
                raise ArchiveError('备份复核版本不连续，未恢复',503)
    # get_report checks each bounded decompressed snapshot and its stored hash.
    for row in reports:
        try:
            result=archive.get_report(row['report_id'])
            record=result['review_handoff']
            if result['report_id']!=row['report_id'] or record['trace']['report_id']!=row['report_id']:
                raise ValueError('report identity mismatch')
            allowed={'report'}|{item['review_item_key'] for item in record['review_items']}|{event['target_key'] for event in record.get('manual_annotation_history',[])}
            if any(event['target_key'] not in allowed for event in record['review_history']):
                raise ValueError('review target mismatch')
        except (KeyError,TypeError,ValueError):
            raise ArchiveError('备份快照身份或复核对象不一致，未恢复',503) from None
    return {'cases':cases,'reports':len(reports),'reviews':reviews,'all_snapshot_hashes_verified':True}


def restore_archive(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source.drive.upper()!='D:' or destination.drive.upper()!='D:':
        raise ArchiveError('备份与恢复目标必须位于D盘')
    if not source.is_file():raise ArchiveError('备份文件不存在')
    if destination.exists() or any(Path(str(destination)+suffix).exists() for suffix in ['-wal','-shm','-journal']):
        raise ArchiveError('恢复目标或其日志已存在；请指定新的数据库文件，不覆盖现有归档',409)
    with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True,timeout=10)) as db:
        size=db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
        CaseArchive(destination).require_space(size)
        destination.parent.mkdir(parents=True,exist_ok=True)
        handle,scratch=tempfile.mkstemp(prefix='.archive_restore_',suffix='.sqlite3',dir=destination.parent)
        os.close(handle);scratch=Path(scratch)
        try:
            with closing(sqlite3.connect(scratch)) as copied:db.backup(copied)
            receipt=validate_archive(scratch)
            # An atomic, exclusive publish: a concurrently created target is never overwritten.
            os.link(scratch,destination)
            return {**receipt,'source':str(source),'destination':str(destination),'existing_archive_overwritten':False}
        except FileExistsError:
            raise ArchiveError('恢复目标已被其他任务创建，未覆盖',409) from None
        finally:
            scratch.unlink(missing_ok=True)
            for suffix in ['-wal','-shm','-journal']:Path(str(scratch)+suffix).unlink(missing_ok=True)
