#  Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
#  All rights reserved.
#
#  This code is licensed under the terms of the 3-clause BSD license
#  (https://opensource.org/license/bsd-3-clause).

from ._bindings import _DSC_TRACE_FILE, _dsc_traces_record, _dsc_insert_trace, _dsc_dump_traces, _DscTracePhase
from .context import _get_ctx
from time import perf_counter
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler
import socketserver


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


@contextmanager
def profile(serve: bool = True):
    start_recording()
    try:
        yield
    finally:
        stop_recording(serve=serve)


def trace(name: str, cat: str = 'python'):
    def _decorator(func):
        def _wrapper(*args, **kwargs):
            _dsc_insert_trace(_get_ctx(), name, cat, int(perf_counter() * 1e6) ,_DscTracePhase.BEGIN)
            res = func(*args, **kwargs)
            _dsc_insert_trace(_get_ctx(), name, cat, int(perf_counter() * 1e6), _DscTracePhase.END)
            return res
        return _wrapper
    return _decorator