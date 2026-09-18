"""Generate text with the trained ApnaAI model (from-scratch, no API).

Usage:
  python generate.py --prompt "Once upon a time" --tokens 200
"""
import argparse
import os

import torch

from config import ModelConfig
from model import GPT
from tokenizer import BPETokenizer

HERE = os.path.dirname(__file__)


def load_model():
    ckpt_path = os.path.join(HERE, "checkpoints", "model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mc = ModelConfig(**ckpt["model_config"])
    model = GPT(mc)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer().load(os.path.join(HERE, "data", "tokenizer.json"))
    return model, tok, ckpt


def generate(model, tok, prompt, max_new_tokens=200, temperature=0.8,
             top_k=40):
    if prompt:
        ids = tok.encode(prompt)
    else:
        ids = [tok.encode("\\n")[0]]
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens, temperature=temperature,
                         top_k=top_k)
    return tok.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    model, tok, ckpt = load_model()
    print(f"[loaded checkpoint from iter {ckpt['iter']}, "
          f"val loss {ckpt['val_loss']:.4f}]\\n")
    text = generate(model, tok, args.prompt, args.tokens,
                    args.temperature, args.top_k)
    print(text)


if __name__ == "__main__":
    main()
