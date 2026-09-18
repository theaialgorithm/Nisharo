"""Build the training data for the Nisharo model.

Steps (all local, no API):
  1. Read the raw English text files in ./data/*.txt
  2. Strip Project Gutenberg license headers/footers so the model only
     learns from the actual prose.
  3. Train the from-scratch BPE tokenizer on the cleaned text.
  4. Encode the whole corpus to token ids and save train/val binary shards.
"""
import glob
import os
import numpy as np

from config import VOCAB_SIZE
from tokenizer import BPETokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def strip_gutenberg(text):
    """Remove the standard Gutenberg boilerplate around the real text."""
    start_markers = ["*** START OF THE PROJECT GUTENBERG",
                     "*** START OF THIS PROJECT GUTENBERG"]
    end_markers = ["*** END OF THE PROJECT GUTENBERG",
                   "*** END OF THIS PROJECT GUTENBERG"]
    for m in start_markers:
        i = text.find(m)
        if i != -1:
            text = text[text.find("\n", i) + 1:]
            break
    for m in end_markers:
        i = text.find(m)
        if i != -1:
            text = text[:i]
            break
    return text.strip()


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    assert files, "No .txt files found in data/. Run the corpus download first."

    parts = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        cleaned = strip_gutenberg(raw)
        parts.append(cleaned)
        print(f"  loaded {os.path.basename(path):24s} {len(cleaned):>9,d} chars")
    text = "\n\n".join(parts)
    print(f"Total corpus: {len(text):,} characters")

    print("Training BPE tokenizer (from scratch)...")
    tok = BPETokenizer()
    tok.train(text, VOCAB_SIZE, verbose=True)
    tok_path = os.path.join(DATA_DIR, "tokenizer.json")
    tok.save(tok_path)
    print(f"Tokenizer saved -> {tok_path} (vocab size {len(tok)})")

    print("Encoding corpus to token ids...")
    ids = tok.encode(text)
    ids = np.array(ids, dtype=np.uint16)
    n = len(ids)
    split = int(n * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))
    print(f"Tokens: {n:,} total  ({len(train_ids):,} train / "
          f"{len(val_ids):,} val)")
    print(f"Compression: {len(text) / n:.2f} chars per token")


if __name__ == "__main__":
    main()
