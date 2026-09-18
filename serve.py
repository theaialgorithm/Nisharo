"""Local HTTP inference server for the Nisharo instruction-finetuned model.

Loads the fine-tuned checkpoint and serves both raw generation and
instruction-formatted chat endpoints.

  python serve.py                       # listens on http://127.0.0.1:8008
  curl -s -X POST http://127.0.0.1:8008/chat \
       -H 'Content-Type: application/json' \
       -d '{"message": "What is gravity?"}'
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from model import GPT
from config import ModelConfig
from tokenizer import BPETokenizer
from generate import generate

torch.set_num_threads(min(16, os.cpu_count() or 8))

HERE = os.path.dirname(os.path.abspath(__file__))

# Load fine-tuned model if available, else base model
CKPT_DIR = os.path.join(HERE, "checkpoints")
PERSONA = os.path.join(CKPT_DIR, "model_persona.pt")
FINETUNED = os.path.join(CKPT_DIR, "model_finetuned.pt")
BASE = os.path.join(CKPT_DIR, "model.pt")

# Prefer the persona-tuned checkpoint (greetings/identity/chit-chat), then the
# instruction fine-tune, then the base pretrained model.
if os.path.exists(PERSONA):
    ckpt_path = PERSONA
elif os.path.exists(FINETUNED):
    ckpt_path = FINETUNED
else:
    ckpt_path = BASE
print(f"Loading checkpoint: {os.path.basename(ckpt_path)}")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
CFG = ModelConfig()
MODEL = GPT(CFG)
MODEL.load_state_dict(ckpt["model"])
MODEL.eval()

TOK = BPETokenizer()
TOK.load(os.path.join(HERE, "data", "tokenizer.json"))

IS_FINETUNED = ckpt_path in (PERSONA, FINETUNED)
print(f"Model loaded | {'instruction-finetuned' if IS_FINETUNED else 'base'} | iter {ckpt.get('iter', '?')} | val loss {ckpt.get('val_loss', 0):.4f}")


def chat_generate(message, max_tokens=200, temperature=0.7, top_k=50):
    """Format as instruction prompt and extract assistant response."""
    prompt = f"<|user|> {message}\n<|assistant|>"
    full = generate(MODEL, TOK, prompt, max_new_tokens=max_tokens,
                    temperature=temperature, top_k=top_k)
    # Extract only assistant part
    if "<|assistant|>" in full:
        response = full.split("<|assistant|>")[-1]
    else:
        response = full[len(prompt):]
    # Trim at end token
    if "<|end|>" in response:
        response = response.split("<|end|>")[0]
    if "<|user|>" in response:
        response = response.split("<|user|>")[0]
    return response.strip()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {
                "status": "ok",
                "model": "instruction-finetuned" if IS_FINETUNED else "base",
                "iter": ckpt.get("iter", 0),
                "val_loss": round(ckpt.get("val_loss", 0), 4)
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
            return

        if self.path == "/chat":
            message = payload.get("message", "")
            if not message:
                self._send(400, {"error": "message required"})
                return
            response = chat_generate(
                message,
                max_tokens=int(payload.get("max_tokens", 200)),
                temperature=float(payload.get("temperature", 0.7)),
                top_k=int(payload.get("top_k", 50)),
            )
            self._send(200, {"message": message, "response": response})

        elif self.path == "/generate":
            prompt = payload.get("prompt", "")
            text = generate(
                MODEL, TOK, prompt,
                max_new_tokens=int(payload.get("max_new_tokens", 150)),
                temperature=float(payload.get("temperature", 0.8)),
                top_k=int(payload.get("top_k", 40)),
            )
            self._send(200, {"prompt": prompt, "completion": text})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *args):
        pass


def main():
    port = int(os.environ.get("PORT", 8008))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving Nisharo on http://127.0.0.1:{port}")
    print(f"  POST /chat     - instruction chat")
    print(f"  POST /generate - raw text generation")
    print(f"  GET  /health   - model info")
    srv.serve_forever()


if __name__ == "__main__":
    main()
