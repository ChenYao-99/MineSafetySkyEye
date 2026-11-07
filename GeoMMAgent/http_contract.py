"""Shared JSON transport failures; field validation remains with each service."""
def install_json_errors(app):
    from flask import jsonify,request
    from werkzeug.exceptions import HTTPException

    @app.errorhandler(413)
    def oversized_request(error):
        limit = app.config.get('MAX_CONTENT_LENGTH', 0)
        return jsonify({'error': '请求超过服务大小限制，请裁剪影像或缩小共同分析范围后重试',
                        'status_code': 413, 'max_request_bytes': limit}), 413

    @app.errorhandler(Exception)
    def safe_api_failure(error):
        if not request.path.startswith('/api/'):
            if isinstance(error,HTTPException):return error
            raise error
        if isinstance(error,HTTPException):
            return jsonify({'error':'接口请求未完成，请核对请求路径与内容','status_code':error.code}),error.code
        app.logger.error('API failure: %s',type(error).__name__)
        return jsonify({'error':'本次任务未完成，未返回成功结果；请检查本地服务日志后重试','status_code':500}),500


def validate_analysis_fields(data):
    """Reject ambiguous primitive types before a proxy, inference or cloud call."""
    for name in ('question', 'query', 'region', 'model_tag', 'channel_order', 'registration','analysis_grid',
                 'image', 'image_t1', 'image_t2', 'report_id', 'case_id'):
        if name in data and not isinstance(data[name], str):
            return name + ' 必须是字符串'
    for name in ('fuse', 'ds_reject', 'nir_synth'):
        if name in data and not isinstance(data[name], bool):
            return name + ' 必须是布尔值'
    for name in ('num', 'start', 'sample_index'):
        value = data.get(name)
        if name == 'sample_index' and value is None:
            continue
        if name in data and (isinstance(value, bool) or not isinstance(value, int)
                             or value < (1 if name == 'num' else 0)):
            return name + ' 必须是' + ('正整数' if name == 'num' else '非负整数')
    return None
