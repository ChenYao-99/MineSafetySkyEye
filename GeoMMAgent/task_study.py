"""Local objective task study. Human/self-reported and automation results stay separate."""
from datetime import datetime,timezone
from pathlib import Path
import copy,hashlib,json,math,random,threading,time,uuid
from case_archive import CaseArchive,ArchiveError
from evidence_reporting import objective_facts
from review_handoff import build_review_record


def now():return datetime.now(timezone.utc).isoformat()


class TaskStudy:
    def __init__(self,bank_path,directory,clock=time.monotonic):
        self.bank_path=Path(bank_path).resolve();self.directory=Path(directory).resolve()
        if self.directory.drive.upper()!='D:':raise ValueError('任务记录必须位于D盘')
        self.bank=json.loads(self.bank_path.read_text(encoding='utf-8'))
        self.tasks={task['id']:task for task in self.bank['tasks']}
        self.bank_sha256=hashlib.sha256(self.bank_path.read_bytes()).hexdigest()
        self.clock=clock;self.boot_id=uuid.uuid4().hex;self.lock=threading.RLock()
        self.directory.mkdir(parents=True,exist_ok=True);self.archive=CaseArchive(self.directory/'task_archive.sqlite3')

    def path(self,token):
        if not isinstance(token,str) or len(token)!=32 or any(c not in '0123456789abcdef' for c in token):raise ArchiveError('任务编号无效')
        return self.directory/(token+'.json')

    def read(self,token):
        path=self.path(token)
        if not path.exists():raise ArchiveError('任务记录不存在',404)
        return json.loads(path.read_text(encoding='utf-8'))

    def write(self,state):
        path=self.path(state['token']);temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
        temporary.replace(path)

    def create(self,data):
        kind=data.get('participant_kind');role=data.get('role')
        if kind not in ('human_self_reported','automated_test') or not isinstance(role,str) or not 1<=len(role.strip())<=80:
            raise ArchiveError('请明确实际参与方式与角色')
        if kind=='human_self_reported' and data.get('actual_operator_confirmed') is not True:raise ArchiveError('真实任务需要实际操作者确认')
        token=uuid.uuid4().hex;ids=list(self.tasks);random.Random(42).shuffle(ids)
        order=[{'task_id':task_id,'arm':arm} for i,task_id in enumerate(ids) for arm in (['assisted','worksheet'] if i%2==0 else ['worksheet','assisted'])]
        state={'token':token,'created_at':now(),'participant_kind':kind,'role_self_reported':role.strip(),
            'participant_identity_verified':False,'professional_or_field_trial':False,'bank_sha256':self.bank_sha256,
            'order':order,'index':0,'trials':[],'current':None,'status':'ready',
            'limits':['objective facts and handoff only;not geological judgement','repeated-case learning possible','single participant not population evidence']}
        with self.lock:self.write(state)
        return {'token':token,'trials':len(order),'participant_kind':kind}

    def current(self,token):
        with self.lock:
            state=self.read(token)
            if state['bank_sha256']!=self.bank_sha256:raise ArchiveError('任务数据已改变，不能继续',409)
            if state['index']==len(state['order']):return {'complete':True,'summary':self.summary(state)}
            assignment=state['order'][state['index']];task=self.tasks[assignment['task_id']]
            if state['current'] is None:
                state['current']={'nonce':uuid.uuid4().hex,'started_at':now(),'started_clock':self.clock(),'boot_id':self.boot_id,'phase':'answer'}
                self.write(state)
            current=state['current']
            if current['boot_id']!=self.boot_id:raise ArchiveError('服务重启打断了本次计时，请另开任务，不计作已完成',409)
            expected=task['expected'];fields=copy.deepcopy(expected) if assignment['arm']=='assisted' else {key:'' for key in expected}
            image=task['detection'].get('rgb_b64') or task['detection'].get('t2_rgb_b64')
            return {'complete':False,'number':state['index']+1,'total':len(state['order']),'nonce':current['nonce'],
                'arm':assignment['arm'],'case_name':task['name'],'evidence':expected,'image_b64':image,
                'image_shape':task['detection']['size'],'initial_fields':fields,'phase':current['phase'],
                'saved_record':current.get('saved_record'),'participant_kind':state['participant_kind']}

    def submit(self,token,data):
        with self.lock:
            state=self.read(token);current=state['current']
            if not current or current['phase']!='answer' or data.get('nonce')!=current['nonce']:raise ArchiveError('任务版本已改变，不能重复提交',409)
            if current['boot_id']!=self.boot_id:raise ArchiveError('计时被服务重启打断',409)
            assignment=state['order'][state['index']];task=self.tasks[assignment['task_id']]
            fields=data.get('fields');note=data.get('note')
            if not isinstance(fields,dict) or set(fields)!=set(task['expected']) or not isinstance(note,str) or not note.strip() or len(note)>1000:
                raise ArchiveError('请完成交接字段及你的核查说明')
            # Validate both arms' location boxes against the same immutable input.
            from manual_annotations import location_box
            report_id=uuid.uuid4().hex;det=copy.deepcopy(task['detection'])
            try:annotation=location_box(data.get('box_px'),det,report_id)
            except ValueError as error:raise ArchiveError(str(error)) from None
            grades={}
            for key,expected in task['expected'].items():
                entered=fields[key]
                if key=='measurement':
                    try:number=float(entered);grades[key]=not isinstance(entered,bool) and math.isfinite(number) and abs(number-expected)<=.01
                    except (TypeError,ValueError):grades[key]=False
                elif key=='capture_dates':grades[key]=isinstance(entered,str) and entered.replace(' ','').lower()==expected.replace(' ','').lower()
                else:grades[key]=isinstance(entered,str) and entered.strip().lower()==expected.strip().lower()
            record={'fields_entered':fields,'review_note':note.strip(),'annotation':annotation,
                'source_task':task['id'],'source_raw_image_sha256':task['raw_image_sha256'],'accident_or_stability_confirmed':False}
            if assignment['arm']=='assisted':
                result={'report_id':report_id,'generated_at':now(),'region':task['region'],'detection':det,'final_answer':'冻结历史结果的真实任务核对',
                    'objective_facts':objective_facts(det),'generation':{'status':'fallback','source':'frozen_task_replay'},'knowledge':[],'agents':[]}
                result['review_handoff']=build_review_record(result)
                self.archive.save_report(result)
                saved=self.archive.submit_manual_annotation(report_id,{'expected_revision':0,'box_px':data['box_px'],
                    'reviewer':'任务参与者 '+token[:8],'evidence':'已冻结公开开发案例的同源原图显示；参与者角色自填未核验','note':note.strip()})
                actual=saved['review_handoff']['manual_annotations'][0]['manual_annotation']
                record['annotation']=actual;record['report_id']=report_id
                record['snapshot_sha256']=saved['archive']['snapshot_sha256']
            else:
                flat=self.directory/(token+'_'+str(state['index'])+'_worksheet.json')
                flat.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
                record=json.loads(flat.read_text(encoding='utf-8'))
            current.update(phase='verify',saved_record=record,grades=grades,submitted_at=now())
            self.write(state)
            return {'saved_record':record,'phase':'verify','nonce':current['nonce']}

    def confirm(self,token,data):
        with self.lock:
            state=self.read(token);current=state['current']
            if not current or current['phase']!='verify' or data.get('nonce')!=current['nonce'] or data.get('saved_record_checked') is not True:
                raise ArchiveError('请先核对已保存记录，不能跳过交接步骤',409)
            if current['boot_id']!=self.boot_id:raise ArchiveError('计时被服务重启打断',409)
            assignment=state['order'][state['index']];elapsed=self.clock()-current['started_clock']
            saved=current['saved_record']
            if assignment['arm']=='assisted':
                reopened=self.archive.get_report(saved['report_id'])
                locations=reopened['review_handoff']['manual_annotations']
                if len(locations)!=1:raise ArchiveError('已保存定位记录不完整，未计作完成',503)
                location=locations[0]
                if location['manual_annotation']!=saved['annotation'] or location['note']!=saved['review_note']:
                    raise ArchiveError('已保存交接记录不一致，未计作完成',503)
            else:
                flat=self.directory/(token+'_'+str(state['index'])+'_worksheet.json')
                if json.loads(flat.read_text(encoding='utf-8'))!=saved:raise ArchiveError('文件交接记录不一致，未计作完成',503)
            state['trials'].append({**assignment,'started_at':current['started_at'],'completed_at':now(),'elapsed_seconds':elapsed,
                'objective_field_grades':current['grades'],'incorrect_or_missing_fields':sum(not value for value in current['grades'].values()),
                'source_bound_location_and_reopened_note_completed':True,'record':current['saved_record'],
                'human_time_observation':state['participant_kind']=='human_self_reported','geological_correctness_scored':False})
            state['index']+=1;state['current']=None;state['status']='complete' if state['index']==len(state['order']) else 'ready'
            self.write(state)
            return {'completed_trials':state['index'],'complete':state['status']=='complete'}

    def summary(self,state):
        import statistics
        by_arm={}
        for arm in ['worksheet','assisted']:
            rows=[row for row in state['trials'] if row['arm']==arm]
            by_arm[arm]={'tasks_completed':len(rows),'incorrect_or_missing_fields':sum(row['incorrect_or_missing_fields'] for row in rows),
                'median_total_task_seconds':statistics.median(row['elapsed_seconds'] for row in rows) if rows else None}
        return {'participant_kind':state['participant_kind'],'status':state['status'],'arms':by_arm,
            'human_self_reported_observations':len(state['trials']) if state['participant_kind']=='human_self_reported' else 0,
            'verified_professional_or_field_trial':False,'repeated_case_learning_possible':True,
            'automatic_prefill_is_deterministic_fact_transfer_not_AI_accuracy':True}
