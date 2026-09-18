"""Native LoRA and answer-preserving SFT for this project's own GPT only."""
import copy
import hashlib
import re
from pathlib import Path

import torch
from torch import nn
from config import ModelConfig
from model import GPT
from tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'experiments/lora_sft_v1'
SEED = 20260918
IGNORE = -1  # Existing GPT.forward uses ignore_index=-1, not HF's -100.
END = '<|end|>'  # Existing multi-token marker; no vocabulary/embedding change.
FIXED_PROMPTS = [
    'hii',
    'Hello! How are you?',
    'What is 5 - 2?',
    'What is machine learning?',
    'Write a Python function to reverse a string.',
]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize(text):
    return re.sub(r'\s+', ' ', text.casefold()).strip()


def prompt_text(question):
    return f'<|user|> {question}\n<|assistant|>'


def encode_pair(tok, question, answer, block_size=256):
    """Right-pad only at collation. Preserve ALL answer and terminator tokens.

    Reject overlong answers; remove old prompt tokens first. Data preparation
    rejects truncated prompts too, because losing a question's context harms
    quality; this primitive nevertheless supports safe answer-preserving cuts.
    """
    p = tok.encode(prompt_text(question))
    a = tok.encode(' ' + answer + END)
    if not question.strip() or not answer.strip() or len(a) > block_size:
        return None
    room = block_size + 1 - len(a)
    if room < len(tok.encode('<|assistant|>')):
        return None
    truncated = len(p) > room
    p = p[-room:]
    ids = p + a
    labels = [IGNORE] * len(p) + a
    return {'x': ids[:-1], 'y': labels[1:], 'prompt_truncated': truncated,
            'answer_tokens': len(a)}


def collate(rows):
    length = max(len(r['x']) for r in rows)
    x = torch.zeros((len(rows), length), dtype=torch.long)
    y = torch.full_like(x, IGNORE)
    for i, row in enumerate(rows):
        n = len(row['x'])
        x[i, :n] = torch.tensor(row['x'])
        y[i, :n] = torch.tensor(row['y'])
    # Token 0 is input filler, NOT a new special token. Causal attention means
    # right-side filler cannot affect earlier supervised positions.
    return x, y


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError('LoRA requires actual nn.Linear projections')
        self.base = base
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5 ** .5)
        self.base.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + ((x @ self.a.T) @ self.b.T) * self.scale

    def merged(self):
        out = copy.deepcopy(self.base)
        with torch.no_grad():
            out.weight.add_((self.b @ self.a) * self.scale)
        return out


def add_lora(model, rank=8, alpha=16):
    model.requires_grad_(False)
    names = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.startswith('transformer.h.'):
            parent_name, leaf = name.rsplit('.', 1)
            setattr(model.get_submodule(parent_name), leaf, LoRALinear(module, rank, alpha))
            names.append(name)
    assert len(names) == model.cfg.n_layer * 4
    return names


def merge_lora(model):
    merged = copy.deepcopy(model)
    for name, module in list(merged.named_modules()):
        if isinstance(module, LoRALinear):
            parent, leaf = name.rsplit('.', 1)
            setattr(merged.get_submodule(parent), leaf, module.merged())
    return merged


def adapter_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_original(path=None):
    path = path or ROOT / 'checkpoints/model_finetuned.pt'
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = GPT(ModelConfig(**ckpt['model_config']))
    model.load_state_dict(ckpt['model'], strict=True)
    return model, BPETokenizer().load(ROOT / 'data/tokenizer.json'), ckpt


@torch.no_grad()
def evaluate(model, rows, batch_size=4):
    model.eval()
    nll = tokens = num_correct = 0
    for start in range(0, len(rows), batch_size):
        x, y = collate(rows[start:start + batch_size])
        logits, loss = model(x, y)
        mask = y != IGNORE
        count = int(mask.sum())
        nll += loss.item() * count
        tokens += count
        num_correct += int(((logits.argmax(-1) == y) & mask).sum())
    return {'assistant_token_nll': nll / tokens,
            'assistant_token_accuracy': num_correct / tokens, 'supervised_tokens': tokens,
            'examples': len(rows)}


@torch.no_grad()
def reply(model, tok, question, max_tokens=80):
    """Fixed greedy decoding, same context and stopping for both arms."""
    model.eval()
    ids = tok.encode(prompt_text(question))
    x = torch.tensor([ids], dtype=torch.long)
    generated = []
    for _ in range(max_tokens):
        logits, _ = model(x[:, -model.cfg.block_size:])
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        generated.append(int(nxt.item()))
        x = torch.cat((x, nxt), dim=1)
        if any(m in tok.decode(generated) for m in [END, '<|user|>']):
            break
    raw = tok.decode(generated)
    answer = raw.split(END)[0].split('<|user|>')[0].strip()
    return {'prompt': question, 'raw_completion': raw, 'response': answer,
            'generated_tokens': len(generated)}
