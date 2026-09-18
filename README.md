# Nisharo — A From-Scratch Language Model

**Nisharo** is a GPT-style language model built entirely from scratch in PyTorch. No pretrained weights, no external AI APIs, no Llama — every component is written and trained from the ground up.

## Architecture

| Component | Details |
|-----------|---------|
| Type | Decoder-only Transformer (GPT-style) |
| Parameters | **13.82M** |
| Layers | 6 |
| Attention Heads | 6 |
| Embedding Dim | 384 |
| Context Length | 256 tokens |
| Vocabulary | 8,000 BPE tokens |
| Tokenizer | Custom byte-level BPE (from scratch) |

## What's Built From Scratch

- **Tokenizer** (`tokenizer.py`) — Byte-level BPE tokenizer trained on the corpus, no HuggingFace tokenizers
- **Model** (`model.py`) — Causal self-attention, MLP blocks, positional embeddings, weight tying, all hand-written
- **Training** (`train.py`) — Cosine LR schedule, gradient clipping, AdamW, checkpoint saving
- **Data Pipeline** (`prepare_data.py`) — Corpus loading, Gutenberg cleanup, tokenization, train/val split
- **Instruction Fine-Tuning** (`fine_tune.py`, `sft_core.py`) — SFT with assistant-only loss masking
- **Persona Training** (`experiments/persona_v1/`) — Identity, greetings, feelings, capabilities
- **Inference Server** (`serve.py`) — HTTP API for chat and generation
- **Generation** (`generate.py`, `generate_chat.py`) — Sampling with temperature and top-k

## Project Structure

```
Nisharo/
├── model.py                 # GPT transformer (from scratch)
├── config.py                # Model and training hyperparameters
├── tokenizer.py             # Byte-level BPE tokenizer (from scratch)
├── train.py                 # Pre-training on English literature corpus
├── prepare_data.py          # Corpus → tokenizer + binary shards
├── fine_tune.py             # Instruction fine-tuning (OpenHermes streaming)
├── sft_core.py              # SFT utilities, LoRA, evaluation
├── prepare_sft.py           # Dataset download, filtering, split freezing
├── serve.py                 # HTTP inference server (chat + generation)
├── generate.py              # CLI text generation (base model)
├── generate_chat.py         # CLI instruction chat (fine-tuned model)
├── requirements.txt         # Dependencies (torch, numpy, regex, requests)
├── data/
│   └── tokenizer.json       # Trained BPE merge rules (8K vocab)
├── checkpoints/             # Model weights (not in repo — train locally)
│   ├── model.pt             # Base pre-trained checkpoint
│   ├── model_finetuned.pt   # Instruction fine-tuned checkpoint
│   └── model_persona.pt     # Persona-tuned checkpoint (latest)
└── experiments/
    ├── persona_v1/          # Persona/identity fine-tuning experiment
    │   ├── persona_data.py  # Dataset generator (785 curated examples)
    │   ├── train_persona.py # Training script
    │   ├── persona.jsonl    # Generated persona dataset
    │   └── results.json     # Training metrics and eval results
    └── lora_sft_v1/         # LoRA SFT experiment (Dolly + OASST1)
        ├── EXPERIMENT.md    # Full experiment documentation
        ├── config.json      # Experiment configuration
        ├── results.json     # Training metrics
        └── train.jsonl      # Filtered instruction dataset (1005 examples)
```

## Training Pipeline

### 1. Pre-Training (Base Model)

Trained on ~5.5M characters of classic English literature (Shakespeare, Sherlock Holmes, Pride and Prejudice, Moby Dick, Frankenstein, Alice in Wonderland, etc.):

```bash
# Download corpus (Project Gutenberg public domain texts)
# Place .txt files in data/

# Build tokenizer and binary data
python prepare_data.py

# Train the base model (~50 min on CPU)
python train.py
# → checkpoints/model.pt
```

### 2. Instruction Fine-Tuning

Fine-tuned on curated Q&A data from OpenHermes-2.5 (streamed, not bulk downloaded):

```bash
python fine_tune.py
# → checkpoints/model_finetuned.pt
```

### 3. Persona Fine-Tuning (Current Best)

Added 785 curated greeting/identity/feelings/capabilities examples + 800 instruction examples for retention:

```bash
# Generate persona dataset
python experiments/persona_v1/persona_data.py

# Fine-tune from instruction checkpoint
python experiments/persona_v1/train_persona.py
# → checkpoints/model_persona.pt
```

## Running the Model

### HTTP Server

```bash
python serve.py
# Serves on http://127.0.0.1:8008
```

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Instruction-formatted chat |
| `/generate` | POST | Raw text generation |
| `/health` | GET | Model info and status |

**Example:**

```bash
curl -X POST http://127.0.0.1:8008/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Hi", "temperature": 0.7}'

# → {"message": "Hi", "response": "Hello there! I'm Nisharo, your AI assistant. How can I assist you today?"}
```

### CLI Chat

```bash
python generate_chat.py --prompt "What is gravity?" --interactive
```

### CLI Generation (Base Model)

```bash
python generate.py --prompt "Once upon a time" --tokens 200
```

## Current Capabilities

After persona fine-tuning, the model handles:

| Category | Example | Status |
|----------|---------|--------|
| Greetings | "Hi", "Hello", "Hey there" | ✅ Works well |
| Identity | "Who are you?", "What's your name?" | ✅ Works well |
| Feelings | "Do you have feelings?", "How are you?" | ✅ Works well |
| Capabilities | "What can you do?", "Are you smart?" | ✅ Works well |
| Simple Facts | "What is 5 - 2?", "Capital of France?" | ✅ Works well |
| Thanks/Goodbye | "Thank you", "Bye" | ✅ Works well |
| Emotional Support | "I'm sad", "I feel lonely" | ✅ Works well |
| Complex Q&A | "What is machine learning?" | ⚠️ Limited |
| Coding | "Write a Python function..." | ⚠️ Limited |

**Honest Limitations:** This is a 13.82M parameter model trained on CPU. It excels at conversational patterns it has been trained on but struggles with complex factual questions and code generation. Scaling up model size, training data, and compute would improve these areas.

## Requirements

```
torch>=2.0
numpy>=1.24
regex>=2023.0
requests>=2.31
```

```bash
# CPU-only install
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy regex requests
```

## Key Design Decisions

1. **From Scratch** — Every line of model code, tokenizer, and training loop is written here. No pretrained weights are loaded.
2. **BPE Tokenizer** — Custom byte-level BPE trained on the same corpus, not borrowed from GPT-2/HuggingFace.
3. **Assistant-Only Loss** — During SFT, loss is computed only on assistant tokens (not the user prompt), improving instruction following.
4. **Persona Data** — Hand-authored greeting/identity/feelings data, not generated by another AI.
5. **CPU Friendly** — Designed to train on commodity hardware without a GPU. The base model trains in ~50 minutes on a 2-core CPU.

## License

MIT

## Author

[@theaialgorithm](https://github.com/theaialgorithm)
