"""Download pinned public datasets, filter short English pairs and freeze splits.
No LLM/inference API is used. Reruns refuse to overwrite the frozen dataset.
"""
import collections
import gzip
import hashlib
import json
import random
import re
from datetime import datetime, timezone

import requests
from sft_core import RUN, ROOT, SEED, FIXED_PROMPTS, encode_pair, normalize, sha
from tokenizer import BPETokenizer

SOURCES = [
    {'id': 'databricks/databricks-dolly-15k',
     'revision': 'bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a',
     'file': 'databricks-dolly-15k.jsonl', 'license': 'cc-by-sa-3.0'},
    {'id': 'OpenAssistant/oasst1',
     'revision': 'fdf72ae0827c1cda404aff25b6603abec9e3399b',
     'file': '2023-04-12_oasst_ready.messages.jsonl.gz', 'license': 'apache-2.0'},
]


def download(source, filename):
    path = RUN / 'source' / (source['id'].split('/')[-1] + '-' + filename)
    url = f"https://huggingface.co/datasets/{source['id']}/resolve/{source['revision']}/{filename}"
    if not path.exists():
        print('Downloading', url, flush=True)
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        temp = path.with_suffix(path.suffix + '.partial')
        with temp.open('wb') as f:
            for block in r.iter_content(1048576):
                f.write(block)
        temp.rename(path)
    return path, url


def heldout_leak(prompt):
    text = normalize(prompt)
    simple = re.sub(r'[^a-z0-9]+', ' ', text).strip()
    fixed = {re.sub(r'[^a-z0-9]+', ' ', normalize(x)).strip() for x in FIXED_PROMPTS}
    return (simple in fixed or 'machine learning' in text
            or bool(re.search(r'5\s*(?:-|minus)\s*2|five\s+minus\s+two', text))
            or ('reverse' in text and 'string' in text)
            or 'how are you' in text)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / 'source').mkdir(exist_ok=True)
    if (RUN / 'manifest.json').exists():
        raise RuntimeError('Frozen run exists; use a new RUN directory for new data')
    tok = BPETokenizer().load(ROOT / 'data/tokenizer.json')
    counts = collections.Counter()
    rows, seen_prompts, seen_pairs = [], set(), set()

    def accept(source, record_id, question, answer, category):
        counts[source + ':considered'] += 1
        question = re.sub(r'\[\d+\]', '', question).strip().replace('\r\n', '\n')
        answer = re.sub(r'\[\d+\]', '', answer).strip().replace('\r\n', '\n')
        text = question + answer
        reason = None
        if not question or not answer or len(question) > 800 or len(answer) > 700:
            reason = 'length'
        elif sum(ord(c) < 128 for c in text) / len(text) < .97:
            reason = 'non_ascii_language_heuristic'
        elif any(s in text.lower() for s in ['<|', 'http://', 'https://', 'as an ai language model', 'as a language model']):
            reason = 'format_or_boilerplate'
        elif text.count('```') % 2 or any(s in text.lower() for s in ['roleplay', 'act as a', 'openassistant']):
            reason = 'unbalanced_code_or_persona'
        elif heldout_leak(question) or heldout_leak(answer):
            reason = 'heldout_prompt_or_topic'
        ph = hashlib.sha256(normalize(question).encode()).hexdigest()
        pairhash = hashlib.sha256((normalize(question) + '\n' + normalize(answer)).encode()).hexdigest()
        encoded = None if reason else encode_pair(tok, question, answer)
        if not reason and (encoded is None or encoded['prompt_truncated'] or len(encoded['x']) > 192 or encoded['answer_tokens'] > 112):
            reason = 'token_budget_complete_answer'
        if not reason and (ph in seen_prompts or pairhash in seen_pairs):
            reason = 'duplicate'
        if reason:
            counts[source + ':rejected:' + reason] += 1
            return
        seen_prompts.add(ph)
        seen_pairs.add(pairhash)
        bucket = int(hashlib.sha256(f'{SEED}:{ph}'.encode()).hexdigest(), 16) % 100
        rows.append({'source': source, 'record_id': str(record_id), 'category': category,
                     'prompt': question, 'answer': answer, 'prompt_sha256': ph,
                     'pair_sha256': pairhash, 'split': 'validation' if bucket < 10 else 'train',
                     **encoded})
        counts[source + ':accepted_before_cap'] += 1

    for src in SOURCES:
        meta = requests.get('https://huggingface.co/api/datasets/' + src['id'] + '/revision/' + src['revision'], timeout=30)
        meta.raise_for_status()
        assert meta.json().get('cardData', {}).get('license') == src['license']
        card, _ = download(src, 'README.md')
        src['card_sha256'] = sha(card)
        if src['id'] == 'OpenAssistant/oasst1':
            license_path, _ = download(src, 'LICENSE')
            src['license_file_sha256'] = sha(license_path)
        path, url = download(src, src['file'])
        src.update(url=url, downloaded_sha256=sha(path), bytes=path.stat().st_size)
        if src['id'].startswith('databricks/'):
            for i, line in enumerate(path.open()):
                r = json.loads(line)
                if r.get('category') not in {'open_qa', 'general_qa', 'closed_qa', 'classification', 'information_extraction', 'brainstorming'}:
                    continue
                q = r['instruction'] + ('\nContext: ' + r['context'] if r.get('context') else '')
                accept(src['id'], i, q, r['response'], r['category'])
        else:
            with gzip.open(path, 'rt') as f:
                messages = {r['message_id']: r for r in map(json.loads, f)}
            for r in messages.values():
                p = messages.get(r.get('parent_id'))
                if not p or p.get('parent_id') is not None or r.get('role') != 'assistant' or p.get('role') != 'prompter':
                    continue  # only context-complete single-turn root pairs
                if any(x.get('lang') != 'en' or x.get('deleted') or x.get('synthetic') or x.get('review_result') is False for x in [p, r]):
                    continue
                if r.get('rank') != 0 or r.get('model_name'):
                    continue
                labels = {name: entry['value'] for name, entry in (r.get('labels') or {}).items()}
                if (labels.get('quality', 0) < .5 or labels.get('toxicity', 0) > .2
                        or any(labels.get(k, 0) > 0 for k in ['spam', 'fails_task', 'pii', 'not_appropriate', 'hate_speech'])):
                    continue
                accept(src['id'], r['message_id'], p['text'], r['text'], 'root_rank0_quality_ge_0.5')
    # Per-source deterministic cap after split: diversity without tail-order bias.
    rng = random.Random(SEED)
    final = []
    for src in SOURCES:
        for split, cap in [('train', 700), ('validation', 32)]:
            pool = sorted([r for r in rows if r['source'] == src['id'] and r['split'] == split], key=lambda r: r['pair_sha256'])
            rng.shuffle(pool)
            final.extend(pool[:cap])
    assert all(not heldout_leak(r['prompt']) for r in final)
    for split in ['train', 'validation']:
        selected = sorted([r for r in final if r['split'] == split], key=lambda r: r['pair_sha256'])
        assert selected, f'Empty {split}'
        with (RUN / f'{split}.jsonl').open('w') as f:
            for row in selected:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {'seed': SEED, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'sources': SOURCES, 'filters': dict(counts),
                'counts': dict(collections.Counter(r['split'] + ':' + r['source'] for r in final)),
                'tokenizer_sha256': sha(ROOT / 'data/tokenizer.json'),
                'fixed_prompts': FIXED_PROMPTS,
                'split_rule': 'SHA256(seed:normalized_prompt_hash) modulo100 <10 => validation; capped per source after split',
                'max_total_tokens': 192, 'max_answer_tokens_including_end': 112,
                'long_prompt_policy': 'reject after answer-preserving encoding; never cut an answer',
                'leakage_note': 'Exact normalized prompts and specified topic variants excluded from new pool. Historical training data unavailable; no historical-unseen claim.',
                'files': {s: sha(RUN / f'{s}.jsonl') for s in ['train', 'validation']}}
    (RUN / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
