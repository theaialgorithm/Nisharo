"""Generate answers from the instruction-fine-tuned Nisharo model.

This loads the fine-tuned checkpoint (not the base pre-trained one) and
formats prompts in the instruction format the model was trained on.

Usage:
  python generate_chat.py --prompt "What is machine learning?"
  python generate_chat.py --prompt "Write a Python function to reverse a string"
"""
import argparse
import os

import torch

from config import ModelConfig
from model import GPT
from tokenizer import BPETokenizer

HERE = os.path.dirname(__file__)

USER_TOKEN = "<|user|>"
ASST_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"


def load_finetuned_model():
    ckpt_path = os.path.join(HERE, "checkpoints", "model_finetuned.pt")
    if not os.path.exists(ckpt_path):
        print("No fine-tuned checkpoint found. Run fine_tune.py first.")
        print("Falling back to base model...")
        ckpt_path = os.path.join(HERE, "checkpoints", "model.pt")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = ModelConfig(**ckpt["model_config"])
    model = GPT(mc)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer().load(os.path.join(HERE, "data", "tokenizer.json"))

    model_type = ckpt.get("type", "base")
    print(f"[loaded {model_type} model | iter {ckpt['iter']} | val loss {ckpt['val_loss']:.4f}]")
    return model, tok


def chat(model, tok, question, max_new_tokens=200, temperature=0.7, top_k=50):
    """Format as instruction and generate answer."""
    prompt = f"{USER_TOKEN} {question}\n{ASST_TOKEN}"
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    full_text = tok.decode(out[0].tolist())

    # Extract just the assistant's response
    if ASST_TOKEN in full_text:
        response = full_text.split(ASST_TOKEN)[-1].strip()
    else:
        response = full_text[len(prompt):].strip()

    # Stop at end token if present
    if END_TOKEN in response:
        response = response[:response.index(END_TOKEN)].strip()

    return response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="What is artificial intelligence?")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    args = ap.parse_args()

    model, tok = load_finetuned_model()

    if args.interactive:
        print("\nNisharo Chat (type 'quit' to exit)")
        print("=" * 40)
        while True:
            try:
                q = input("\nYou: ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if not q:
                    continue
                answer = chat(model, tok, q, args.tokens, args.temperature, args.top_k)
                print(f"\nNisharo: {answer}")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nBye!")
    else:
        answer = chat(model, tok, args.prompt, args.tokens, args.temperature, args.top_k)
        print(f"\nQuestion: {args.prompt}")
        print(f"\nNisharo: {answer}")


if __name__ == "__main__":
    main()
