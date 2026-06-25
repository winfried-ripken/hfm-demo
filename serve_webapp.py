#!/usr/bin/env python3
"""Tiny static server for the client-side MD web app.

    python serve_webapp.py [port]

Then open http://localhost:8000/ . Serves from the `webapp/` directory with the
right MIME types for ES modules and the .onnx model.
"""
import sys
import http.server
import socketserver
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
ROOT = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".onnx": "application/octet-stream",
        ".wasm": "application/wasm",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving {ROOT} at http://localhost:{PORT}/  (Ctrl+C to stop)")
    httpd.serve_forever()
