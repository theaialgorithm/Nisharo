# ApnaAI - a language model built entirely from scratch

This is a small GPT-style language model implemented **from scratch in PyTorch**.
There are **no third-party AI / LLM APIs** anywhere in this project - not for the
tokenizer, not for the training data, and not for inference. Every part of the
model is defined in plain Python in this folder and trained locally on this
machine.

## What is included

| File | Purpose |
|------|---------|
| `tokenizer.py`  | Byte-level BPE tokenizer written from scratch. Learns its merges from the corpus. |
| `model.py`      | The GPT transformer: embeddings, causal multi-head self-attention, MLP blocks, layer norm, LM head, sampling. |
| `config.py`     | Model + training hyper-parameters. |
| `prepare_data.py` | Cleans the local text corpus, trains the tokenizer, encodes to binary shards. |
| `train.py`      | The training loop (AdamW, cosine LR schedule, gradient clipping, checkpointing). |
| `generate.py`   | Load the trained checkpoint and generate text from a prompt. |
| `serve.py`      | Tiny local HTTP inference server (standard library only). |
| `data/`         | Public-domain English text (Project Gutenberg) - the training corpus. |
| `checkpoints/`  | Saved model weights (`model.pt`). |

## How it was built

1. **Corpus** - several public-domain English books plus TinyShakespeare
   (~5.4M characters) downloaded over plain HTTP. No API involved.
2. **Tokenizer** - a byte-pair-encoding tokenizer trained from scratch on that
   corpus (vocabulary of 8,000 tokens).
3. **Model** - a ~14M parameter GPT (6 layers, 6 heads, 384-dim embeddings,
   256-token context).
4. **Training** - trained on CPU using PyTorch's MKL backend (16 threads).

## Training Results

| Iter | Train Loss | Val Loss | Time |
|------|-----------|---------|------|
| 0 | 9.06 | 9.06 | 1 min |
| 500 | 4.92 | 5.04 | 22 min |
| 1000 | 4.37 | 4.65 | 42 min |
| 1500 | 4.09 | 4.47 | 60 min |
| 2000 | 3.89 | 4.40 | 79 min |
| 2500 | 3.81 | 4.35 | 98 min |
| **3000** | **3.73** | **4.35** | **116 min** |

## Reproduce it

```bash
pip install -r requirements.txt
python prepare_data.py     # build tokenizer + train/val shards
python train.py            # train the model (writes checkpoints/model.pt)
python generate.py --prompt "It was a bright morning" --tokens 200
python serve.py            # optional: local HTTP inference server
```

## Architecture

```
GPT (14M parameters)
  - Token Embedding (8000 x 384)
  - Positional Embedding (256 x 384)
  - 6x Transformer Block:
      - LayerNorm -> Multi-Head Causal Self-Attention (6 heads)
      - LayerNorm -> MLP (384 -> 1536 -> 384, GELU)
  - LayerNorm
  - LM Head (384 -> 8000, weight-tied)
```

## Next Steps

- **Instruction Fine-tuning**: Train on Q&A pairs so the model answers questions instead of just completing text.
- **More Data**: Add more English text corpora for better language understanding.
- **Scale Up**: With GPU access, increase model size (more layers, wider embeddings).
