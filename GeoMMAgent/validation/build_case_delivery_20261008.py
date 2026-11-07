"""One real development case, current single/paired handoff, exact inputs and local ZIP."""
from datetime import datetime
from pathlib import Path
import argparse,base64,hashlib,json,os,shutil,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];ML=Path('D:/LMHLD');sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
import psutil,requests
OUT=ROOT/'cache/case_delivery_20261008';BASE='http://127.0.0.1:5200'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1)
    attempt=parser.parse_args().attempt;assert 1<=attempt<=3
    if attempt>1:OUT=OUT.with_name(OUT.name+'_v'+str(attempt))
    assert not OUT.exists(),'Do not overwrite a completed source-bound delivery'
    assert not any(c.status=='LISTEN' and c.laddr.port in [5200,5201] for c in psutil.net_connections(kind='tcp'))
    OUT.mkdir();inputs=OUT/'inputs';inputs.mkdir()
    source=ML/'datasets/mine_benchmark_20261002/development/LasCruces'
    for epoch in ['pre','post']:shutil.copy2(source/(epoch+'_BGRN_10m.tif'),inputs/(epoch+'_BGRN_10m.tif'))
    protocol={'registered_at':datetime.now().astimezone().isoformat(),'case':'LasCruces','role':'already-observed development demonstration',
        'tasks':['single post epoch','paired pre/post'],'input_sha256':{epoch:sha(inputs/(epoch+'_BGRN_10m.tif')) for epoch in ['pre','post']},
        'model_tag':'mine','threshold':.4,'analysis_grid':'native','local_facts_only':True,
        'manual_boxes':'fixed fractions of original image dimensions;internal workflow demonstration,not expert annotation',
        'new_unseen_mine_or_field_effectiveness_evaluation':False,'new_CNN_fits_or_threshold_search_or_cloud_calls':0,
        'reference_original_inputs_copied_without_resampling':True}
    write(OUT/'protocol.json',protocol)
    env=os.environ.copy();env.update(KAT_CASE_ARCHIVE_PATH=str(OUT/'case_archive.sqlite3'),PYTHONIOENCODING='utf-8')
    session=requests.Session();session.trust_env=False
    records=[]
    try:
        with (OUT/'startup.log').open('wb') as log:
            result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'start_demo.ps1'),
                '-NoBrowser','-LocalFactsOnly','-BackendPort','5200','-DetectorPort','5201'],env=env,stdout=log,stderr=subprocess.STDOUT,
                timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
        assert result.returncode==0,'See startup.log'
        health=session.get(BASE+'/api/health',timeout=10).json();assert health['reporting_mode']=='local_facts_only'
        case=session.post(BASE+'/api/cases',json={'mine_name':'公开历史案例：LasCruces','title':'同次结果与补记位置的离线交接',
            'notes':'内部演示数据，不是专业用户意见、现场试用或新增独立矿区测试'},timeout=10).json()
        payload={epoch:base64.b64encode((inputs/(epoch+'_BGRN_10m.tif')).read_bytes()).decode() for epoch in ['pre','post']}
        for name,endpoint,data in [('single','pipeline_upload',{'image':payload['post']}),
            ('paired','pipeline_compare',{'image_t1':payload['pre'],'image_t2':payload['post']})]:
            response=session.post(BASE+'/api/'+endpoint,json={**data,'model_tag':'mine','case_id':case['case_id']},timeout=180)
            assert response.status_code==200,response.text[:400];original=response.json();rid=original['report_id']
            h,w=original['detection']['size']
            response=session.post(BASE+'/api/archive/reports/'+rid+'/manual_annotations',json={
                'expected_revision':0,'box_px':[w*.2,h*.25,w*.38,h*.4],'reviewer':'内部演示流程验收',
                'evidence':'同次原图显示；固定比例定位框仅用于验证交接功能',
                'note':'内部演示补记，不是事故边界或专家意见；实际原因须另行核查'},timeout=15)
            assert response.status_code==200,response.text[:400];reviewed=response.json()
            assert reviewed['objective_facts']==original['objective_facts']
            assert reviewed['archive']['snapshot_sha256']==original['archive']['snapshot_sha256']
            folder=OUT/name;folder.mkdir();write(folder/'archived_result.json',reviewed)
            for fmt in ['json','html']:
                response=session.post(BASE+'/api/review_record',json={'report_id':rid,'format':fmt},timeout=30)
                assert response.status_code==200,response.text[:400];export=response.json()
                content=base64.b64decode(export['content_b64']);(folder/('review.'+fmt)).write_bytes(content)
                if fmt=='html':
                    text=content.decode('utf-8');assert '<svg id="scene"' in text and '人工定位框' in text
                    assert '10.45%' in text and '不是本页案例的精度' in text
                else:
                    record=json.loads(content);assert len(record['manual_annotations'])==1
                    assert not record['project_field_trial_or_expert_endorsement']
            records.append({'kind':name,'report_id':rid,'snapshot_sha256':original['archive']['snapshot_sha256'],
                'request_id':original['detection']['request_id'],'weight_sha256':original['review_handoff']['trace']['weights_sha256'],
                'facts':original['objective_facts'],'manual_location_count':1,'human_or_expert_evidence':False})
            print('DELIVERED real raw input and current offline review',name,flush=True)
        # Include declared original provenance without bundling unrelated papers or credentials.
        provenance={}
        for name in ['acquisition.json','label_reference.json']:
            path=source/name
            if path.is_file():provenance[name]=read(path)
        write(OUT/'source_provenance.json',provenance)
        note='''# 矿安天眼：公开历史案例的本地复核交付

先用浏览器打开 single/review.html 或 paired/review.html。原图、候选叠加、人工补记、来源与修改记录均在同一文件中；离线查看不依赖Python、GPU或云端。

这是已参与迭代的 LasCruces 开发案例，不是新的盲测。定位框和意见由内部演示程序生成，不是专家审核或现场试用。模型的 CL0002 独立失败结果单列展示，不能当作本案例精度。定位框不是事故边界，面积不加入模型统计。

inputs/ 保存本次原始前后期四波段影像，未经重采样。source_provenance.json 保留原项目中的来源说明；各来源使用条件应分别保留。报告中的PNG是显示版本，不是新观测或精度更高的数据。

复核数值与来源可对照 review.json 和 manifest.json。若要重新执行检测，需要项目的两端Python环境和同SHA-256默认权重；本ZIP不包含解释器、模型权重或凭据，不是另一台净机安装证明。原始输入可通过正常单期/双期上传入口重跑，生成的新结果具有新的请求与报告编号。

本包是本地工程审阅产物，未上传或发布；基础分割模型及GeoMMAgent等第三方组件的归属与许可见原项目声明。
'''
        (OUT/'README.md').write_text(note,encoding='utf-8')
        selected=['README.md','protocol.json','source_provenance.json','inputs/pre_BGRN_10m.tif','inputs/post_BGRN_10m.tif',
            'single/review.html','single/review.json','paired/review.html','paired/review.json']
        manifest={'created_at':datetime.now().astimezone().isoformat(),'files':{name:{'sha256':sha(OUT/name),'bytes':(OUT/name).stat().st_size} for name in selected},
            'records':records,'credentials_or_models_or_unrelated_datasets_bundled':False,'local_only_not_published':True}
        write(OUT/'manifest.json',manifest)
        archive=OUT/'矿安天眼_公开案例离线交接.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zip:
            for name in selected+['manifest.json']:zip.write(OUT/name,name)
        with zipfile.ZipFile(archive) as zip:assert zip.testzip() is None
        write(OUT/'verification.json',{'passed':True,'records':records,'actual_GPU_tasks_on_one_development_mine':2,
            'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'same_input_fact_source_and_manual_note_preserved':True,
            'source_roles_and_independent_failure_limits_visible':True,'new_unseen_mine_predictions_or_cloud_calls_or_model_fits':0,
            'production_archive_written':False,'runtime_identity':{'backend':health['backend_runtime_sha256']}})
    finally:
        subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'stop_demo.ps1'),
            '-BackendPort','5200','-DetectorPort','5201'],env=env,capture_output=True,timeout=40,creationflags=subprocess.CREATE_NO_WINDOW)


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');main()
