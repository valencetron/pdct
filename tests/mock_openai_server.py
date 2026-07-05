"""Minimal OpenAI-compatible mock server for configure/doctor smoke tests.

Serves /v1/models and /v1/chat/completions with deterministic, schema-valid
responses good enough to pass the capability probe and doctor stage 6.
Usable as a pytest fixture (import start_server) or standalone:

    python3 tests/mock_openai_server.py [port]
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

DISTILL_JSON = {
    "title": "pgvector chosen for retrieval",
    "summary": "Team benchmarked pgvector against a dedicated vector "
               "database and chose pgvector for operational simplicity, "
               "using an HNSW index.",
    "concepts": ["pgvector", "vector-database", "hnsw", "benchmarking",
                 "retrieval"],
}
JUDGE_JSON = {"score": 9, "rationale": "The note directly answers the question."}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send({"object": "list",
                        "data": [{"id": "mock-model", "object": "model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            req = {}
        text = json.dumps(req.get("messages", []))
        # judge-style prompts want a bare verdict; distill wants the schema
        if "judge" in text.lower() or '"score"' in text:
            content = json.dumps(JUDGE_JSON)
        else:
            content = json.dumps(DISTILL_JSON)
        self._send({
            "id": "chatcmpl-mock", "object": "chat.completion",
            "model": req.get("model", "mock-model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
        })


def start_server(port: int = 0) -> tuple[HTTPServer, int, threading.Thread]:
    srv = HTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, srv.server_address[1], t


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8399
    srv, port, t = start_server(p)
    print(f"mock openai server on http://127.0.0.1:{port}/v1")
    try:
        t.join()
    except KeyboardInterrupt:
        srv.shutdown()
