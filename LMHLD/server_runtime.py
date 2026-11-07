"""Bounded Windows-compatible WSGI serving with project-local temporary buffers."""
import os


def serve_app(app, port, debug=False):
    host = os.environ.get('KAT_BIND_HOST', '127.0.0.1')
    if debug:
        app.config['KAT_HTTP_RUNTIME'] = 'flask_debug'
        app.run(host=host, port=port, debug=True, use_reloader=True)
        return
    from waitress import serve
    app.config['KAT_HTTP_RUNTIME'] = 'waitress'
    serve(app, host=host, port=port, threads=8, connection_limit=64,
          inbuf_overflow=1024*1024, outbuf_overflow=1024*1024,
          max_request_body_size=64*1024*1024, channel_timeout=180,
          ident='kat', expose_tracebacks=False)
