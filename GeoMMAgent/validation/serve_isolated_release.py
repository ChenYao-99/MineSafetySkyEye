"""Run the copied backend in its new environment with cloud calls disabled."""
import argparse,json,os,sys
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--release',required=True);parser.add_argument('--db',required=True);args=parser.parse_args()
root=Path(args.release).resolve();os.environ['KAT_LMHLD_DIR']=str(root/'LMHLD');os.environ['DETECTOR_PORT']='5111'
os.environ['YOLO_AUTOINSTALL']='false';os.environ['PYTHONIOENCODING']='utf-8'
sys.path.insert(0,str(root/'GeoMMAgent'))
import backend as service
from case_archive import CaseArchive
from flask import jsonify
initialised=service.coordinator is not None
service.coordinator=None;service.llm_fast_check=lambda *args,**kwargs:False
service.case_archive=CaseArchive(args.db)


@service.app.route('/api/deployment_identity')
def identity():
    import flask,numpy,PIL,camel
    return jsonify({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'source':service.__file__,
      'module_paths':{'flask':flask.__file__,'numpy':numpy.__file__,'PIL':PIL.__file__,'camel':camel.__file__},
      'initial_coordinator_loaded':initialised,'cloud_disabled_for_acceptance':True,
      'case_database':str(service.case_archive.path),'field_or_independent_model_acceptance':False})


service.app.run(host='127.0.0.1',port=5110,debug=False,threaded=True)
