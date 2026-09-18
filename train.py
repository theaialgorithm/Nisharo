"""Train the ApnaAI GPT model from scratch on the local English corpus.

CPU-only friendly: uses all available cores, small model by default. Run
`python prepare_data.py` first to build data/train.bin, data/val.bin and
data/tokenizer.json.
"""
import math
import os
import time

import numpy as np
import torch

from config import ModelConfig, TrainConfig
from model import GPT
from tokenizer import BPETokenizer

HERE = os.path.dirname(__file__)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64))
                     for i in ix])
    y = torch.stack([torch.from_numpy(
        data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def get_lr(it, tc):
    if it < tc.warmup_iters:
        return tc.learning_rate * (it + 1) / (tc.warmup_iters + 1)
    if it > tc.lr_decay_iters:
        return tc.min_lr
    ratio = (it - tc.warmup_iters) / (tc.lr_decay_iters - tc.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return tc.min_lr + coeff * (tc.learning_rate - tc.min_lr)


@torch.no_grad()
def estimate_loss(model, splits, tc, mc, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(tc.eval_iters)
        for k in range(tc.eval_iters):
            xb, yb = get_batch(data, mc.block_size, tc.batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    torch.manual_seed(1337)
    n_threads = min(16, os.cpu_count() or 8)
    torch.set_num_threads(n_threads)
    device = "cpu"
    print(f"Training on CPU with {n_threads} threads")

    tc = TrainConfig()
    data_dir = os.path.join(HERE, tc.data_dir)

    tok = BPETokenizer().load(os.path.join(data_dir, "tokenizer.json"))
    mc = ModelConfig(vocab_size=len(tok))

    train_data = np.memmap(os.path.join(data_dir, "train.bin"),
                           dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "val.bin"),
                         dtype=np.uint16, mode="r")
    splits = {"train": train_data, "val": val_data}

    model = GPT(mc).to(device)
    n_params = model.num_params()
    print(f"Model: {n_params / 1e6:.2f}M parameters | vocab {mc.vocab_size} | "
          f"{mc.n_layer}L {mc.n_head}H {mc.n_embd}D | ctx {mc.block_size}")

    optimizer = model.configure_optimizers(
        tc.weight_decay, tc.learning_rate, (tc.beta1, tc.beta2))

    os.makedirs(os.path.join(HERE, tc.checkpoint_dir), exist_ok=True)
    ckpt_path = os.path.join(HERE, tc.checkpoint_dir, "model.pt")

    best_val = float("inf")
    t0 = time.time()
    for it in range(tc.max_iters + 1):
        lr = get_lr(it, tc)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % tc.eval_interval == 0:
            losses = estimate_loss(model, splits, tc, mc, device)
            dt = time.time() - t0
            print(f"iter {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | lr {lr:.2e} | {dt/60:.1f} min",
                  flush=True)
            if losses["val"] < best_val or tc.always_save_checkpoint:
                best_val = min(best_val, losses["val"])
                torch.save({
                    "model": model.state_dict(),
                    "model_config": mc.__dict__,
                    "iter": it,
                    "val_loss": losses["val"],
                }, ckpt_path)

        xb, yb = get_batch(train_data, mc.block_size, tc.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()

        if it % tc.log_interval == 0 and it % tc.eval_interval != 0:
            dt = time.time() - t0
            print(f"iter {it:5d} | train batch loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | {dt/60:.1f} min", flush=True)

    print(f"Done. Best val loss {best_val:.4f}. Checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
