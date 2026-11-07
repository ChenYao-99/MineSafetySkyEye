"""Real detector proxy with local grounded fallback and an isolated archive."""
from pathlib import Path
import argparse,sys,threading
ROOT=Path('D:/GeoMMAgent');sys.path.insert(0,str(ROOT))
import backend as service
from case_archive import CaseArchive
from flask import jsonify
from werkzeug.serving import make_server
service.coordinator=None
service.llm_fast_check=lambda *args,**kwargs:False
server=None


@service.app.route('/api/local_e2e_shutdown',methods=['POST'])
def shutdown():
    threading.Thread(target=server.shutdown,daemon=True).start();return jsonify({'status':'stopping'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--db',required=True);parser.add_argument('--port',type=int,default=5105);args=parser.parse_args()
    service.case_archive=CaseArchive(args.db)
    server=make_server('127.0.0.1',args.port,service.app,threaded=True)
    timer=threading.Timer(300,server.shutdown);timer.daemon=True;timer.start()
    try:server.serve_forever()
    finally:timer.cancel();server.server_close()
