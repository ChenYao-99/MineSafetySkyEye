"""Immutable analysis snapshots and append-only local review records on D:."""
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import hashlib
import json
import sqlite3
import uuid
import zlib
import shutil
from manual_annotations import decode as decode_manual,encode as encode_manual

REVIEW_LABELS = {'needs_followup':'疑似异常需跟进', 'likely_work_change':'可能为作业变化',
                'observation_issue':'观测条件可能影响结果', 'insufficient_evidence':'证据不足，暂无法判断',
                'withdrawn':'撤回此前意见'}


class ArchiveError(Exception):
    def __init__(self, message, status=400, **details):
        super().__init__(message); self.status=status; self.details=details


def now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def text(value, name, maximum, required=True):
    if not isinstance(value,str) or len(value)>maximum or (required and not value.strip()):
        raise ArchiveError(f'{name}须为'+('非空' if required else '')+f'文本，最多{maximum}字')
    return value.strip()


def revision(value):
    if isinstance(value,bool) or not isinstance(value,int) or value<0:
        raise ArchiveError('请提供当前记录的 expected_revision（非负整数）')
    return value


def public_snapshot(value):
    """Persist display/result data, not credentials, raw inputs or provider diagnostics."""
    if isinstance(value,dict):
        blocked={'api_key','authorization','password','token','secret','image_path',
                 'case_context','archive','review_history'}
        return {str(k):public_snapshot(v) for k,v in value.items()
                if not str(k).startswith('_') and str(k).lower() not in blocked
                and not (str(k) in ('image','image_t1','image_t2') and isinstance(v,str))}
    if isinstance(value,(list,tuple)):return [public_snapshot(v) for v in value]
    return value


class CaseArchive:
    def __init__(self,path):
        self.path=Path(path).resolve()
        if self.path.drive.upper()!='D:':raise ValueError('案例归档数据库必须在D盘')

    def require_space(self,extra_bytes):
        existing=self.path.parent
        while not existing.exists():existing=existing.parent
        if shutil.disk_usage(existing).free < extra_bytes+64*1024*1024:
            raise ArchiveError('D盘空间不足，未写入新归档或创建备份；请先释放空间',503)

    @contextmanager
    def connection(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        db=sqlite3.connect(str(self.path),timeout=10,isolation_level=None)
        db.row_factory=sqlite3.Row
        try:
            version=db.execute('PRAGMA user_version').fetchone()[0]
            tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            if version not in (0,1) or (version==0 and tables):
                raise ArchiveError('归档数据库版本不兼容，未改写现有数据',503)
            if version==1:
                required={
                    'cases':{'case_id','mine_name','title','notes','created_at','updated_at','revision'},
                    'reports':{'report_id','case_id','created_at','kind','model_tag','capture_t1','capture_t2','payload','payload_sha256','review_revision'},
                    'reviews':{'event_id','report_id','revision','target_key','status','reviewer','evidence','note','created_at'},
                    'case_history':{'case_id','revision','metadata','created_at'}}
                if not set(required)<=tables or any(not fields<={r[1] for r in db.execute('PRAGMA table_info('+name+')')} for name,fields in required.items()):
                    raise ArchiveError('归档数据库结构不兼容，未创建或修补缺失表',503)
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS cases(
                    case_id TEXT PRIMARY KEY,mine_name TEXT NOT NULL,title TEXT NOT NULL,notes TEXT NOT NULL,
                    created_at TEXT NOT NULL,updated_at TEXT NOT NULL,revision INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS reports(
                    report_id TEXT PRIMARY KEY,case_id TEXT NOT NULL REFERENCES cases(case_id),
                    created_at TEXT NOT NULL,kind TEXT NOT NULL,model_tag TEXT,
                    capture_t1 TEXT,capture_t2 TEXT,payload BLOB NOT NULL,payload_sha256 TEXT NOT NULL,
                    review_revision INTEGER NOT NULL DEFAULT 0);
                CREATE INDEX IF NOT EXISTS reports_case_time ON reports(case_id,created_at DESC);
                CREATE TABLE IF NOT EXISTS reviews(
                    event_id TEXT PRIMARY KEY,report_id TEXT NOT NULL REFERENCES reports(report_id),
                    revision INTEGER NOT NULL,target_key TEXT NOT NULL,status TEXT NOT NULL,
                    reviewer TEXT NOT NULL,evidence TEXT NOT NULL,note TEXT NOT NULL,created_at TEXT NOT NULL,
                    UNIQUE(report_id,revision));
                CREATE INDEX IF NOT EXISTS reviews_report_revision ON reviews(report_id,revision);
                CREATE TABLE IF NOT EXISTS case_history(
                    case_id TEXT NOT NULL REFERENCES cases(case_id),revision INTEGER NOT NULL,
                    metadata TEXT NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(case_id,revision));
                PRAGMA user_version=1;
            ''')
            yield db
        finally:db.close()

    def create_case(self,mine_name,title,notes=''):
        mine_name=text(mine_name,'矿区名称',120);title=text(title,'案例标题',200);notes=text(notes,'案例备注',4000,False)
        row={'case_id':uuid.uuid4().hex,'mine_name':mine_name,'title':title,'notes':notes,
             'created_at':now(),'updated_at':now(),'revision':0}
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO cases VALUES(:case_id,:mine_name,:title,:notes,:created_at,:updated_at,:revision)',row)
            db.execute('INSERT INTO case_history VALUES(?,?,?,?)',(row['case_id'],0,json.dumps(row,ensure_ascii=False),row['created_at']))
            db.execute('COMMIT')
        return row

    def get_case(self,case_id):
        with self.connection() as db:
            db.execute('BEGIN')
            row=db.execute('SELECT * FROM cases WHERE case_id=?',(case_id,)).fetchone()
            history=[{**dict(item),'metadata':json.loads(item['metadata'])} for item in db.execute('SELECT * FROM case_history WHERE case_id=? ORDER BY revision',(case_id,))]
        if row is None:raise ArchiveError('案例不存在',404)
        return {**dict(row),'metadata_history':history}

    def list_cases(self,query='',limit=50,offset=0):
        query=text(query,'查询',200,False)
        if not 1<=limit<=100 or offset<0:raise ArchiveError('分页范围无效')
        with self.connection() as db:
            rows=db.execute('''SELECT c.*,COUNT(r.report_id) AS report_count FROM cases c
                LEFT JOIN reports r ON r.case_id=c.case_id
                WHERE instr(c.mine_name,?)>0 OR instr(c.title,?)>0
                GROUP BY c.case_id ORDER BY c.updated_at DESC,c.case_id LIMIT ? OFFSET ?''',(query,query,limit,offset)).fetchall()
            total=db.execute('SELECT COUNT(*) FROM cases WHERE instr(mine_name,?)>0 OR instr(title,?)>0',(query,query)).fetchone()[0]
        return {'cases':[dict(row) for row in rows],'total':total,'limit':limit,'offset':offset}

    def edit_case(self,case_id,data):
        expected=revision(data.get('expected_revision'))
        fields={name:text(data.get(name),name,maximum,name!='notes') for name,maximum in [('mine_name',120),('title',200),('notes',4000)]}
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT revision FROM cases WHERE case_id=?',(case_id,)).fetchone()
            if row is None:raise ArchiveError('案例不存在',404)
            if row[0]!=expected:raise ArchiveError('案例已更新，请刷新后再保存',409,current_revision=row[0])
            db.execute('UPDATE cases SET mine_name=?,title=?,notes=?,updated_at=?,revision=revision+1 WHERE case_id=?',
                       (fields['mine_name'],fields['title'],fields['notes'],now(),case_id))
            db.execute('INSERT INTO case_history VALUES(?,?,?,?)',(case_id,expected+1,json.dumps(fields,ensure_ascii=False),now()))
            db.execute('COMMIT')
        return self.get_case(case_id)

    def save_report(self,result,case_id=None):
        snapshot=public_snapshot(copy.deepcopy(result))
        for generation in (snapshot.get('generation'),snapshot.get('review_handoff',{}).get('automatic_analysis',{}).get('generation')):
            if isinstance(generation,dict):generation.pop('reason',None)
        content=json.dumps(snapshot,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
        if len(content)>128*1024*1024:raise ArchiveError('分析结果过大，无法归档；请缩小分析范围',413)
        digest=hashlib.sha256(content).hexdigest();record=snapshot['review_handoff']
        compressed=zlib.compress(content,6)
        self.require_space(len(compressed)*2)
        inputs=record.get('input_evidence') or {};t1=inputs.get('t1') or inputs.get('image') or {};t2=inputs.get('t2') or {}
        if not case_id:
            case_id=self.create_case('未标注矿区','分析 '+str(result['generated_at']))['case_id']
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if not db.execute('SELECT 1 FROM cases WHERE case_id=?',(case_id,)).fetchone():raise ArchiveError('归档案例不存在',404)
            db.execute('''INSERT INTO reports(report_id,case_id,created_at,kind,model_tag,capture_t1,capture_t2,payload,payload_sha256)
                VALUES(?,?,?,?,?,?,?,?,?)''',(snapshot['report_id'],case_id,snapshot['generated_at'],record['summary']['kind'],
                  record['trace'].get('model_tag'),t1.get('acquisition_datetime'),t2.get('acquisition_datetime'),compressed,digest))
            db.execute('UPDATE cases SET updated_at=?,revision=revision+1 WHERE case_id=?',(now(),case_id))
            db.execute('COMMIT')
        return self.get_report(snapshot['report_id'])

    def list_reports(self,case_id,limit=50,offset=0):
        case=self.get_case(case_id)
        if not 1<=limit<=100 or offset<0:raise ArchiveError('分页范围无效')
        with self.connection() as db:
            rows=db.execute('''SELECT report_id,case_id,created_at,kind,model_tag,capture_t1,capture_t2,review_revision,payload_sha256
                FROM reports WHERE case_id=? ORDER BY created_at DESC,report_id DESC LIMIT ? OFFSET ?''',(case_id,limit,offset)).fetchall()
            total=db.execute('SELECT COUNT(*) FROM reports WHERE case_id=?',(case_id,)).fetchone()[0]
        return {'case':case,'reports':[dict(row) for row in rows],'total':total,'limit':limit,'offset':offset}

    def get_report(self,report_id):
        with self.connection() as db:
            db.execute('BEGIN')
            row=db.execute('SELECT * FROM reports WHERE report_id=?',(report_id,)).fetchone()
            if row is None:raise ArchiveError('归档分析结果不存在',404)
            events=[dict(e) for e in db.execute('SELECT * FROM reviews WHERE report_id=? ORDER BY revision',(report_id,))]
            case=dict(db.execute('SELECT * FROM cases WHERE case_id=?',(row['case_id'],)).fetchone())
        try:
            decompressor=zlib.decompressobj()
            content=decompressor.decompress(row['payload'],128*1024*1024+1)
            if len(content)>128*1024*1024 or not decompressor.eof:raise ValueError('payload limit')
            if hashlib.sha256(content).hexdigest()!=row['payload_sha256']:raise ValueError('hash mismatch')
            result=json.loads(content)
        except (ValueError,zlib.error):raise ArchiveError('归档完整性校验失败，未返回不可信结果',503) from None
        record=result['review_handoff'];latest={}
        normalized=[]
        for stored_event in events:
            try:event=decode_manual(stored_event,result['detection'],report_id)
            except (ValueError,KeyError,TypeError):raise ArchiveError('人工定位框与归档影像不一致，未返回不可信记录',503) from None
            event['status_label']=REVIEW_LABELS[event['status']]
            event['reviewer_identity_verified']=False
            event['interpretation_is_verified_ground_truth']=False
            event['supersedes_event_id']=(latest.get(event['target_key']) or {}).get('event_id')
            latest[event['target_key']]=event
            normalized.append(event)
        events=normalized
        active=[event for event in latest.values() if event['status']!='withdrawn']
        for item in record['review_items']:
            key=item.get('review_item_key') or 'image:'+str(item.get('id'))
            item['review_item_key']=key
            if key in latest:item['human_review']=latest[key];item['review_status']='human_note_recorded' if latest[key]['status']!='withdrawn' else 'not_reviewed'
        record.update(review_state='human_notes_recorded' if active else 'not_reviewed',
                      accepted_human_review_decisions=active,review_history=events,review_revision=row['review_revision'])
        record['manual_annotations']=[event for event in active if 'manual_annotation' in event]
        record['manual_annotation_history']=[event for event in events if 'manual_annotation' in event]
        record['project_field_trial_or_expert_endorsement']=False
        result['case_context']=case
        result['archive']={'persisted':True,'case_id':case['case_id'],'snapshot_sha256':row['payload_sha256'],
                           'review_revision':row['review_revision'],'schema_version':1}
        return result

    def submit_review(self,report_id,data):
        allowed={'expected_revision','target_key','status','reviewer','evidence','note'}
        if set(data)-allowed:raise ArchiveError('复核只能提交意见字段，不能修改模型量测或来源')
        expected=revision(data.get('expected_revision'))
        target=text(data.get('target_key'),'复核对象',100);status=data.get('status')
        if not isinstance(status,str) or status not in REVIEW_LABELS:raise ArchiveError('复核状态无效')
        reviewer=text(data.get('reviewer'),'复核人',80);evidence=text(data.get('evidence'),'复核依据',2000);note=text(data.get('note'),'判断说明',4000)
        result=self.get_report(report_id);record=result['review_handoff']
        if not record['trace']['trace_complete']:raise ArchiveError('原始请求或权重标识不完整，暂不能提交复核',422)
        targets={'report'}|{row['review_item_key'] for row in record['review_items']}|{row['target_key'] for row in record.get('manual_annotation_history',[])}
        if target not in targets:raise ArchiveError('复核区域不在该报告列出的范围内')
        if status=='withdrawn' and not any(row['target_key']==target for row in record['accepted_human_review_decisions']):
            raise ArchiveError('该对象没有可撤回的有效意见')
        if target.startswith('manual:'):
            previous=next(row for row in reversed(record['manual_annotation_history']) if row['target_key']==target)
            try:note=encode_manual(previous['manual_annotation']['box_px'],note,result['detection'],report_id)
            except ValueError as error:raise ArchiveError(str(error)) from None
        event={'event_id':uuid.uuid4().hex,'report_id':report_id,'revision':expected+1,'target_key':target,
               'status':status,'reviewer':reviewer,'evidence':evidence,'note':note,'created_at':now()}
        return self._store_review_event(event,expected,result['case_context']['case_id'])

    def submit_manual_annotation(self,report_id,data):
        if not isinstance(data,dict) or set(data)-{'expected_revision','box_px','reviewer','evidence','note'}:
            raise ArchiveError('人工补记只能提交定位框和核查意见')
        expected=revision(data.get('expected_revision'))
        reviewer=text(data.get('reviewer'),'复核人',80);evidence=text(data.get('evidence'),'复核依据',2000)
        note=text(data.get('note'),'判断说明',4000)
        result=self.get_report(report_id)
        if not result['review_handoff']['trace']['trace_complete']:raise ArchiveError('同次请求或权重标识不完整，暂不能补记',422)
        try:encoded=encode_manual(data.get('box_px'),note,result['detection'],report_id)
        except ValueError as error:raise ArchiveError(str(error)) from None
        event={'event_id':uuid.uuid4().hex,'report_id':report_id,'revision':expected+1,'target_key':'manual:'+uuid.uuid4().hex,
            'status':'needs_followup','reviewer':reviewer,'evidence':evidence,'note':encoded,'created_at':now()}
        return self._store_review_event(event,expected,result['case_context']['case_id'])

    def _store_review_event(self,event,expected,case_id):
        report_id=event['report_id']
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            current=db.execute('SELECT review_revision FROM reports WHERE report_id=?',(report_id,)).fetchone()
            if current is None:raise ArchiveError('归档分析结果不存在',404)
            if current[0]!=expected:raise ArchiveError('复核记录已更新，请刷新后再提交；旧意见未覆盖新版本',409,current_revision=current[0])
            db.execute('INSERT INTO reviews VALUES(:event_id,:report_id,:revision,:target_key,:status,:reviewer,:evidence,:note,:created_at)',event)
            db.execute('UPDATE reports SET review_revision=review_revision+1 WHERE report_id=?',(report_id,))
            db.execute('UPDATE cases SET updated_at=? WHERE case_id=?',(now(),case_id))
            db.execute('COMMIT')
        return self.get_report(report_id)

    def backup(self):
        size=0
        if self.path.exists():
            source=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)
            try:size=source.execute('PRAGMA page_count').fetchone()[0]*source.execute('PRAGMA page_size').fetchone()[0]
            finally:source.close()
        self.require_space(size)
        directory=self.path.parent/'backups';directory.mkdir(parents=True,exist_ok=True)
        destination=directory/('case_archive_'+uuid.uuid4().hex+'.sqlite3')
        target=sqlite3.connect(str(destination))
        try:
            with self.connection() as source:source.backup(target)
        except Exception:
            target.close();destination.unlink(missing_ok=True);raise
        else:target.close()
        return destination
