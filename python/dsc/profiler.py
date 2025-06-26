#  Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
#  All rights reserved.
#
#  This code is licensed under the terms of the 3-clause BSD license
#  (https://opensource.org/license/bsd-3-clause).

from ._bindings import _DSC_TRACE_FILE, _dsc_tracing_enabled, _dsc_traces_record, _dsc_insert_trace, _dsc_dump_traces, _DscTracePhase
from .context import _get_ctx
from time import perf_counter
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler
import socketserver
from functools import wraps


class _PerfettoServer(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress the output of the HTTP server
        pass

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        return super().end_headers()

    def do_GET(self):
        self.server.last_request = self.path  # pyright: ignore[reportAttributeAccessIssue]
        return super().do_GET()

    def do_POST(self):
        self.send_error(404, 'File not found')


def _serve_traces():
    # Taken from https://github.com/jax-ml/jax
    port = 9001
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('127.0.0.1', port), _PerfettoServer) as httpd:
        url = f'https://ui.perfetto.dev/#!/?url=http://127.0.0.1:{port}/{_DSC_TRACE_FILE}'
        print(f'Open URL in browser: {url}')

        while httpd.__dict__.get('last_request') != '/' + _DSC_TRACE_FILE:
            httpd.handle_request()


def start_recording():
    _dsc_traces_record(_get_ctx(), True)


def stop_recording(dump: bool = True, serve: bool = False):
    _dsc_traces_record(_get_ctx(), False)

    if dump or serve:
        _dsc_dump_traces(_get_ctx())

    if serve:
        _serve_traces()

def _is_tracing_enabled() -> bool:
    # TODO: should I just remove ctx?
    return bool(_dsc_tracing_enabled(None))

def profile(serve: bool = True):
    def _decorator(func):
        if not _is_tracing_enabled():
            return func
        @wraps(func)
        def _wrapper(*args, **kwargs):
            start_recording()
            res = func(*args, **kwargs)
            stop_recording(serve=serve)
            return res
        return _wrapper
    return _decorator


_BEGIN_PHASE = _DscTracePhase.BEGIN.value.encode('ascii')
_END_PHASE = _DscTracePhase.END.value.encode('ascii')

def trace(name: str, cat: str = 'python'):
    def _decorator(func):
        if not _is_tracing_enabled():
            return func

        # Encode name and cat once
        name_ = name.encode('ascii')
        cat_ = cat.encode('ascii')
        @wraps(func)
        def _wrapper(*args, **kwargs):
            _dsc_insert_trace(_get_ctx(), name_, cat_, int(perf_counter() * 1e6), _BEGIN_PHASE)
            res = func(*args, **kwargs)
            _dsc_insert_trace(_get_ctx(), name_, cat_, int(perf_counter() * 1e6), _END_PHASE)
            return res
        return _wrapper
    return _decorator