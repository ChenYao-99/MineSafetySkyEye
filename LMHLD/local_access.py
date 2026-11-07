"""Allow the local UI and CLI while refusing requests from unrelated websites."""
from urllib.parse import urlsplit
import os


def install_local_access(app):
    from flask import request,jsonify
    loopbacks={'127.0.0.1','localhost','::1'}

    def permitted_origin(origin):
        try:
            caller=urlsplit(origin);target=urlsplit(request.host_url)
            host=os.environ.get('KAT_BIND_HOST','127.0.0.1')
            aliases=loopbacks if target.hostname in loopbacks else {host}
            return caller.scheme==target.scheme and caller.hostname in aliases and caller.port==target.port
        except ValueError:return False

    @app.before_request
    def enforce_local_origin():
        host=urlsplit(request.host_url).hostname
        if host not in loopbacks|{os.environ.get('KAT_BIND_HOST','127.0.0.1')}:
            return jsonify({'error':'请求主机与本地部署地址不一致','status_code':403}),403
        origin=request.headers.get('Origin')
        if origin and not permitted_origin(origin):
            return jsonify({'error':'请从本项目启动后的本地网址访问；不接受其他网站或直接文件页面的请求','status_code':403}),403
        if not origin and request.headers.get('Sec-Fetch-Site')=='cross-site':
            return jsonify({'error':'不接受其他网站发起的本地接口请求','status_code':403}),403

    @app.after_request
    def local_cors(response):
        origin=request.headers.get('Origin')
        if origin and permitted_origin(origin):
            response.headers['Access-Control-Allow-Origin']=origin
            response.headers.add('Vary','Origin')
            response.headers['Access-Control-Allow-Methods']='GET, POST, PATCH, OPTIONS'
            response.headers['Access-Control-Allow-Headers']='Content-Type'
        return response
