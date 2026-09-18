"""Minimal local HTTP inference server for the trained ApnaAI model.

Pure standard library (http.server) + the from-scratch model. No frameworks,
no external APIs. Start it and POST prompts to get generated text.

  python serve.py                      # listens on http://127.0.0.1:8008
  curl -s -X POST http://127.0.0.1:8008/generate \
       -H 'Content-Type: application/json' \
       -d '{"prompt": "It was", "max_new_tokens": 120}'
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from generate import load_model, generate

torch.set_num_threads(min(16, os.cpu_count() or 8))
MODEL, TOK, CKPT = load_model()
print(f"Model loaded (iter {CKPT['iter']}, val loss {CKPT['val_loss']:.4f})")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "iter": CKPT["iter"]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
            return
        prompt = payload.get("prompt", "")
        text = generate(
            MODEL, TOK, prompt,
            max_new_tokens=int(payload.get("max_new_tokens", 150)),
            temperature=float(payload.get("temperature", 0.8)),
            top_k=int(payload.get("top_k", 40)),
        )
        self._send(200, {"prompt": prompt, "completion": text})

    def log_message(self, *args):
        pass  # keep the console quiet


def main():
    port = int(os.environ.get("PORT", 8008))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving ApnaAI model on http://127.0.0.1:{port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
