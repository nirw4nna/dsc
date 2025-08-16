#  Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
#  All rights reserved.
#
#  This code is licensed under the terms of the 3-clause BSD license
#  (https://opensource.org/license/bsd-3-clause).

from ._bindings import _dsc_tracing_enabled, _dsc_insert_trace, _dsc_dump_traces
from .context import _get_ctx
from time import perf_counter
from http.server import SimpleHTTPRequestHandler
import socketserver
from functools import wraps
import atexit
import os


PERFETTO = os.getenv('PERFETTO', '0') != '0'

@atexit.register
def _dump_traces():
    # TODO: only if envar is set!
    _dsc_dump_traces(_get_ctx())
    if PERFETTO:
        _serve_traces()


def _is_tracing_enabled() -> bool:
    return bool(_dsc_tracing_enabled())


def trace(name: str):
    def _decorator(func):
        if not _is_tracing_enabled():
            return func

        # Encode name and cat once
        name_ = name.encode('ascii')
        @wraps(func)
        def _wrapper(*args, **kwargs):
            start_us = int(perf_counter() * 1e6)
            res = func(*args, **kwargs)
            end_us = int(perf_counter() * 1e6)
            _dsc_insert_trace(_get_ctx(), name_, start_us, end_us - start_us)
            return res
        return _wrapper
    return _decorator


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
        url = f'https://ui.perfetto.dev/#!/?url=http://127.0.0.1:{port}/traces.json'
        print(f'Open URL in browser: {url}')

        while httpd.__dict__.get('last_request') != '/traces.json':
            httpd.handle_request()
