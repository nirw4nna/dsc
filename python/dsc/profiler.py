#  Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
#  All rights reserved.
#
#  This code is licensed under the terms of the 3-clause BSD license
#  (https://opensource.org/license/bsd-3-clause).

from ._bindings import _dsc_dump_traces, _dsc_traces_record, _dsc_clear_traces
from .context import _get_ctx
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler
import socketserver


def start_recording():
    _dsc_traces_record(_get_ctx(), True)


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


def _serve_traces(traces_file: str):
    # Taken from https://github.com/jax-ml/jax
    port = 9001
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('127.0.0.1', port), _PerfettoServer) as httpd:
        url = f'https://ui.perfetto.dev/#!/?url=http://127.0.0.1:{port}/{traces_file}'
        print(f'Open URL in browser: {url}')

        while httpd.__dict__.get('last_request') != '/' + traces_file:
            httpd.handle_request()


def stop_recording(traces_file: str, clear: bool = True):
    _dsc_traces_record(_get_ctx(), False)
    _dsc_dump_traces(_get_ctx(), traces_file)

    _serve_traces(traces_file)

    if clear:
        _dsc_clear_traces(_get_ctx())


@contextmanager
def profile(dump_file: str = 'traces.json'):
    start_recording()
    try:
        yield
    finally:
        stop_recording(dump_file, True)
