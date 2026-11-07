"""Standalone local task entry; no detector, cloud model or production archive."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project_runtime import configure_runtime_paths
configure_runtime_paths()
from flask import Flask,jsonify,request,send_file
from task_study import TaskStudy
from case_archive import ArchiveError
from local_access import install_local_access
from http_contract import install_json_errors
from server_runtime import serve_app


def create_app(directory=None):
    directory=directory or ROOT/'data/task_studies_20261008'
    store=TaskStudy(ROOT/'cache/high_value_sprint_20261008/task_bank.json',directory)
    app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=64*1024
    install_local_access(app);install_json_errors(app)
    @app.errorhandler(ArchiveError)
    def failure(error):return jsonify({'error':str(error)}),error.status
    @app.before_request
    def body():
        if request.method=='POST' and not isinstance(request.get_json(silent=True),dict):return jsonify({'error':'请求须为JSON对象'}),400
    @app.get('/')
    def page():return send_file(ROOT/'task_study.html')
    @app.get('/api/health')
    def health():return jsonify({'service':'kat-task-study','status':'ok','inference_or_cloud_calls':0,'data_directory':str(store.directory)})
    @app.post('/api/study/sessions')
    def start():return jsonify(store.create(request.get_json()))
    @app.get('/api/study/sessions/<token>/current')
    def current(token):return jsonify(store.current(token))
    @app.post('/api/study/sessions/<token>/submit')
    def submit(token):return jsonify(store.submit(token,request.get_json()))
    @app.post('/api/study/sessions/<token>/confirm')
    def confirm(token):return jsonify(store.confirm(token,request.get_json()))
    @app.get('/api/study/sessions/<token>/result')
    def result(token):return jsonify(store.summary(store.read(token)))
    return app,store


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=5198);parser.add_argument('--directory');args=parser.parse_args()
    app,_=create_app(args.directory)
    serve_app(app,args.port)
