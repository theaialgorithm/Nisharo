"""Instruction fine-tuning for the Nisharo model.

Streams high-quality Q&A data from HuggingFace (OpenHermes-2.5) and fine-tunes
our from-scratch GPT model to follow instructions instead of just completing text.

Key principles:
  - Data quality > quantity: we use curated instruction data, not random text
  - Streaming: data is NOT fully downloaded, it streams during training
  - Fine-tuning only: we start from our pre-trained checkpoint, not from scratch
  - English only
  - No third-party AI APIs
"""
import math
import os
import time
import json

import numpy as np
import torch

from config import ModelConfig, TrainConfig
from model import GPT
from tokenizer import BPETokenizer

HERE = os.path.dirname(__file__)

# Special tokens for instruction format
USER_TOKEN = "<|user|>"
ASST_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"


def load_pretrained():
    """Load the pre-trained base model checkpoint."""
    ckpt_path = os.path.join(HERE, "checkpoints", "model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = ModelConfig(**ckpt["model_config"])
    model = GPT(mc)
    model.load_state_dict(ckpt["model"])
    tok = BPETokenizer().load(os.path.join(HERE, "data", "tokenizer.json"))
    print(f"Loaded pre-trained model (iter {ckpt['iter']}, val loss {ckpt['val_loss']:.4f})")
    print(f"Model: {model.num_params()/1e6:.2f}M params | vocab {mc.vocab_size}")
    return model, tok, mc


def format_conversation(sample):
    """Convert an OpenHermes sample to our instruction format."""
    convs = sample.get("conversations", [])
    if not convs or len(convs) < 2:
        return None

    parts = []
    for turn in convs:
        role = turn.get("from", "")
        value = turn.get("value", "").strip()
        if not value:
            continue
        if role == "human":
            parts.append(f"{USER_TOKEN} {value}")
        elif role == "gpt":
            parts.append(f"{ASST_TOKEN} {value}{END_TOKEN}")

    if len(parts) < 2:
        return None
    return "\n".join(parts)


def stream_instruction_data(max_examples=5000):
    """Stream high-quality instruction data from HuggingFace.

    Uses streaming mode so the full dataset is NOT downloaded to disk.
    """
    from datasets import load_dataset

    print(f"Streaming instruction data from OpenHermes-2.5 (max {max_examples} examples)...")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

    examples = []
    seen = 0
    skipped = 0

    for sample in ds:
        seen += 1
        text = format_conversation(sample)
        if text is None:
            skipped += 1
            continue

        # Quality filters
        if len(text) < 50 or len(text) > 4000:
            skipped += 1
            continue

        examples.append(text)

        if len(examples) % 500 == 0:
            print(f"  collected {len(examples)}/{max_examples} (scanned {seen}, skipped {skipped})")

        if len(examples) >= max_examples:
            break

    print(f"Collected {len(examples)} instruction examples (scanned {seen}, skipped {skipped})")
    return examples


def encode_examples(tok, examples, block_size):
    """Encode instruction examples into training batches."""
    all_ids = []
    for text in examples:
        ids = tok.encode(text)
        if len(ids) > block_size:
            ids = ids[:block_size]
        all_ids.append(ids)
    return all_ids


def get_batch(encoded_data, block_size, batch_size, device):
    """Sample a random batch from the encoded instruction data."""
    batch_x = []
    batch_y = []

    for _ in range(batch_size):
        idx = torch.randint(len(encoded_data), (1,)).item()
        ids = encoded_data[idx]

        if len(ids) < block_size + 1:
            ids = ids + [0] * (block_size + 1 - len(ids))

        x = torch.tensor(ids[:block_size], dtype=torch.long)
        y = torch.tensor(ids[1:block_size + 1], dtype=torch.long)
        batch_x.append(x)
        batch_y.append(y)

    return torch.stack(batch_x).to(device), torch.stack(batch_y).to(device)


def get_lr(it, warmup, max_iters, max_lr, min_lr):
    if it < warmup:
        return max_lr * (it + 1) / (warmup + 1)
    if it > max_iters:
        return min_lr
    ratio = (it - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    torch.manual_seed(42)
    n_threads = min(16, os.cpu_count() or 8)
    torch.set_num_threads(n_threads)
    device = "cpu"
    print(f"Fine-tuning on CPU with {n_threads} threads")
    print("=" * 60)

    MAX_EXAMPLES = 5000
    MAX_ITERS = 2000
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    MIN_LR = 5e-6
    WARMUP_ITERS = 100
    EVAL_INTERVAL = 200
    LOG_INTERVAL = 25
    GRAD_CLIP = 1.0

    model, tok, mc = load_pretrained()
    model.train()

    examples = stream_instruction_data(max_examples=MAX_EXAMPLES)

    split = int(len(examples) * 0.9)
    train_examples = examples[:split]
    val_examples = examples[split:]
    print(f"Split: {len(train_examples)} train / {len(val_examples)} val")

    print("Encoding instruction data...")
    train_encoded = encode_examples(tok, train_examples, mc.block_size)
    val_encoded = encode_examples(tok, val_examples, mc.block_size)
    print(f"Encoded: {len(train_encoded)} train / {len(val_encoded)} val sequences")

    print(f"\nSample instruction (first 200 chars):")
    print(f"  {train_examples[0][:200]}...")
    print()

    optimizer = model.configure_optimizers(0.01, LEARNING_RATE, (0.9, 0.95))

    ckpt_path = os.path.join(HERE, "checkpoints", "model_finetuned.pt")
    best_val = float("inf")
    t0 = time.time()

    print(f"Starting fine-tuning: {MAX_ITERS} iters, batch {BATCH_SIZE}, lr {LEARNING_RATE}")
    print("=" * 60)

    for it in range(MAX_ITERS + 1):
        lr = get_lr(it, WARMUP_ITERS, MAX_ITERS, LEARNING_RATE, MIN_LR)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % EVAL_INTERVAL == 0:
            model.eval()
            val_losses = []
            for _ in range(min(50, len(val_encoded))):
                xb, yb = get_batch(val_encoded, mc.block_size, BATCH_SIZE, device)
                _, loss = model(xb, yb)
                val_losses.append(loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            model.train()

            dt = time.time() - t0
            print(f"iter {it:5d} | val loss {val_loss:.4f} | lr {lr:.2e} | {dt/60:.1f} min",
                  flush=True)

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "model_config": mc.__dict__,
                    "iter": it,
                    "val_loss": val_loss,
                    "type": "instruction-finetuned",
                }, ckpt_path)
                print(f"  -> saved best checkpoint (val {val_loss:.4f})")

        xb, yb = get_batch(train_encoded, mc.block_size, BATCH_SIZE, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if it % LOG_INTERVAL == 0 and it % EVAL_INTERVAL != 0:
            dt = time.time() - t0
            print(f"iter {it:5d} | train loss {loss.item():.4f} | lr {lr:.2e} | {dt/60:.1f} min",
                  flush=True)

    print(f"\nDone. Best val loss {best_val:.4f}. Checkpoint -> {ckpt_path}")
    print("Run: python generate_chat.py --prompt 'What is machine learning?'")


if __name__ == "__main__":
    main()
