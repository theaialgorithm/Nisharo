"""Persona SFT: teach the own-model to greet, state identity, and chit-chat.

Starts from the CURRENTLY SERVING checkpoint (checkpoints/model_finetuned.pt),
fine-tunes the full model on curated persona data mixed with a sample of the
existing instruction data (for retention), and saves to a NEW checkpoint
(checkpoints/model_persona.pt). Original weights are never overwritten.

CPU-only, from-scratch model, no third-party AI/inference API.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sft_core import (  # noqa: E402
    collate, encode_pair, evaluate, load_original, reply, FIXED_PROMPTS,
)

RUN = Path(__file__).resolve().parent
SEED = 20260918
START_CKPT = ROOT / "checkpoints/model_finetuned.pt"
OUT_CKPT = ROOT / "checkpoints/model_persona.pt"
INSTRUCT = ROOT / "experiments/lora_sft_v1/train.jsonl"
INSTRUCT_VAL = ROOT / "experiments/lora_sft_v1/validation.jsonl"
PERSONA = RUN / "persona.jsonl"

BATCH = 16
MAX_STEPS = 800
WARMUP = 30
LR = 2e-4
MIN_LR = 2e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
INSTRUCT_KEEP = 800  # sample of instruction rows mixed in for retention

# Held-out persona prompts (NOT verbatim in training answers) to judge quality.
EVAL_PROMPTS = [
    "Hi", "Hello", "Hey there", "What is your name?", "Who are you?",
    "How are you?", "Do you have feelings?", "What can you do?",
    "Are you smart?", "Thank you", "I'm sad", "What is 5 - 2?",
    "What is the capital of France?", "What is machine learning?",
    "Write a Python function to reverse a string.",
]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def encode_rows(tok, raw, block_size=256):
    rows = []
    for r in raw:
        enc = encode_pair(tok, r["prompt"], r["answer"], block_size)
        if enc is None or enc["prompt_truncated"]:
            continue
        rows.append(enc)
    return rows


def get_lr(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    ratio = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * ratio))


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(SEED)

    print(f"[load] start checkpoint: {START_CKPT.name}", flush=True)
    model, tok, ckpt = load_original(START_CKPT)
    block = model.cfg.block_size

    persona = load_jsonl(PERSONA)
    instruct = load_jsonl(INSTRUCT)
    # sample instruction rows deterministically for retention
    idx = torch.randperm(len(instruct), generator=rng).tolist()[:INSTRUCT_KEEP]
    instruct_sample = [instruct[i] for i in idx]

    train_raw = persona + instruct_sample
    train_rows = encode_rows(tok, train_raw, block)
    perm = torch.randperm(len(train_rows), generator=rng).tolist()
    train_rows = [train_rows[i] for i in perm]

    # Validation: instruction held-out set (assistant-only NLL, comparable arm).
    val_raw = load_jsonl(INSTRUCT_VAL)
    val_rows = encode_rows(tok, val_raw, block)

    print(f"[data] persona={len(persona)} instruct_sample={len(instruct_sample)} "
          f"train_encoded={len(train_rows)} val_encoded={len(val_rows)}", flush=True)

    # Baseline replies + val NLL BEFORE persona training.
    before_val = evaluate(model, val_rows)
    before_replies = {p: reply(model, tok, p)["response"] for p in EVAL_PROMPTS}

    model.train()
    opt = model.configure_optimizers(WEIGHT_DECAY, LR, (0.9, 0.95))

    def batches():
        order = torch.randperm(len(train_rows), generator=rng).tolist()
        for s in range(0, len(order), BATCH):
            yield [train_rows[i] for i in order[s:s + BATCH]]

    step = 0
    t0 = time.time()
    log = []
    while step < MAX_STEPS:
        for batch in batches():
            if step >= MAX_STEPS:
                break
            lr = get_lr(step)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = collate(batch)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if step % 50 == 0 or step == MAX_STEPS - 1:
                dt = time.time() - t0
                msg = (f"step {step:4d}/{MAX_STEPS}  loss {loss.item():.4f}  "
                       f"lr {lr:.2e}  {dt:6.1f}s")
                print(msg, flush=True)
                log.append({"step": step, "loss": loss.item(), "lr": lr, "sec": dt})
            step += 1

    train_secs = time.time() - t0
    after_val = evaluate(model, val_rows)
    after_replies = {p: reply(model, tok, p)["response"] for p in EVAL_PROMPTS}

    # Save NEW checkpoint (same schema as originals for serve.py compatibility).
    out = {
        "model": model.state_dict(),
        "model_config": ckpt["model_config"],
        "note": "persona_v1 full SFT from model_finetuned.pt",
    }
    for k in ("tokenizer", "iter", "train_config"):
        if k in ckpt:
            out[k] = ckpt[k]
    torch.save(out, OUT_CKPT)

    results = {
        "start_checkpoint": str(START_CKPT),
        "out_checkpoint": str(OUT_CKPT),
        "batch": BATCH, "max_steps": MAX_STEPS, "lr": LR, "min_lr": MIN_LR,
        "train_seconds": train_secs,
        "train_examples": len(train_rows), "val_examples": len(val_rows),
        "val_nll_before": before_val["assistant_token_nll"],
        "val_nll_after": after_val["assistant_token_nll"],
        "val_acc_before": before_val["assistant_token_accuracy"],
        "val_acc_after": after_val["assistant_token_accuracy"],
        "log": log,
        "replies_before": before_replies,
        "replies_after": after_replies,
    }
    with open(RUN / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n==== VAL (instruction held-out, assistant-only NLL) ====", flush=True)
    print(f"before {before_val['assistant_token_nll']:.4f} -> "
          f"after {after_val['assistant_token_nll']:.4f}", flush=True)
    print("\n==== REPLIES AFTER PERSONA SFT ====", flush=True)
    for p in EVAL_PROMPTS:
        print(f"\nUSER: {p}\nBEFORE: {before_replies[p]}\nAFTER : {after_replies[p]}",
              flush=True)
    print(f"\nsaved -> {OUT_CKPT}  ({train_secs:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
