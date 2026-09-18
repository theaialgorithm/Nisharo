"""A byte-level Byte-Pair-Encoding (BPE) tokenizer, implemented from scratch.

No external tokenizer library or API is used. The tokenizer learns its merge
rules directly from the training corpus, exactly like GPT-2's tokenizer does,
but every line of it lives in this file so the model is fully self-contained.

Pipeline:
  1. Split raw text into coarse chunks with a GPT-2 style regex (keeps words,
     punctuation and whitespace as separate units so merges stay meaningful).
  2. Encode every chunk as a sequence of raw UTF-8 bytes (0..255).
  3. Repeatedly find the most frequent adjacent byte/token pair across the
     corpus and merge it into a new token, until the target vocab size.
"""
import json
import regex as re
from collections import Counter

# GPT-2 style pre-tokenisation pattern.
SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    def __init__(self):
        # merges: dict[(int, int)] -> int (new token id)
        self.merges = {}
        # vocab: dict[int] -> bytes
        self.vocab = {i: bytes([i]) for i in range(256)}

    # ------------------------------------------------------------------ train
    def train(self, text, vocab_size, verbose=True):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # Count unique chunks so BPE works on word frequencies (fast).
        chunk_freq = Counter(SPLIT_PATTERN.findall(text))
        # Represent each chunk as a list of byte-token ids.
        words = {}
        for chunk, freq in chunk_freq.items():
            ids = tuple(chunk.encode("utf-8"))
            if ids:
                words[ids] = freq

        for i in range(num_merges):
            # Count all adjacent pairs weighted by word frequency.
            pairs = Counter()
            for ids, freq in words.items():
                for a, b in zip(ids, ids[1:]):
                    pairs[(a, b)] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            # Apply the merge to every word.
            words = {self._merge_word(ids, best, new_id): freq
                     for ids, freq in words.items()}
            if verbose and (i + 1) % 500 == 0:
                print(f"  merge {i + 1}/{num_merges}  "
                      f"{self.vocab[best[0]]!r}+{self.vocab[best[1]]!r} "
                      f"(count {pairs[best]})", flush=True)

    @staticmethod
    def _merge_word(ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return tuple(out)

    # ----------------------------------------------------------------- encode
    def _encode_chunk(self, ids):
        ids = list(ids)
        while len(ids) >= 2:
            # Find the pair with the lowest merge id (earliest learned merge).
            pairs = set(zip(ids, ids[1:]))
            candidate = min(
                pairs, key=lambda p: self.merges.get(p, float("inf"))
            )
            if candidate not in self.merges:
                break
            new_id = self.merges[candidate]
            ids = list(self._merge_word(tuple(ids), candidate, new_id))
        return ids

    def encode(self, text):
        out = []
        for chunk in SPLIT_PATTERN.findall(text):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    # ----------------------------------------------------------------- decode
    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------- save / load
    def save(self, path):
        payload = {
            "merges": [[list(k), v] for k, v in self.merges.items()],
            "vocab_size": len(self.vocab),
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    def load(self, path):
        with open(path) as f:
            payload = json.load(f)
        self.merges = {tuple(k): v for k, v in payload["merges"]}
        self.vocab = {i: bytes([i]) for i in range(256)}
        # Rebuild vocab in merge order so byte concatenation is correct.
        for (a, b), new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        return self

    def __len__(self):
        return len(self.vocab)
