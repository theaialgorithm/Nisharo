### Own-model bounded LoRA experiment — September 18, 2026

**Outcome: not a useful chatbot yet. No promotion.** The candidate still fails all five basic content checks. A small held-out token-metric improvement is not evidence of reliable answering, coding ability, or Claude/Gemini-level capability. Serving files and original weights are unchanged; no public deployment, server restart, GitHub push, external base model, AI API, or QLoRA was used.

### Measured experiment

| Item | Result |
|---|---|
| Original | Existing `checkpoints/model_finetuned.pt`, saved iteration 1400 |
| Architecture | Own GPT, 6 layers / 6 heads / width 384 / vocab 8000 / context 256; 13,817,856 base parameters |
| Native LoRA | Rank 8, alpha 16; all 24 attention QKV/output and MLP linear projections; 294,912 trainable parameters (2.13% of base parameter count) |
| Resources | 2 CPU cores; no CUDA; 7.8 GiB system RAM reported; PyTorch 2.14.0+cpu; FP32 |
| Data | 1,005 train = 700 Dolly + 305 OASST1; 63 validation = 32 Dolly + 31 OASST1 |
| Benchmark | One warm-up + six timed optimizer steps; mean 0.7026 seconds/step; estimated 88.52 seconds |
| Actual training | 126 steps / one pass; 91.01 seconds; 54,330 supervised tokens |
| Bound | min(one pass, 240 steps, benchmark-derived 300-second budget), plus 360-second training-loop guard checked between steps |
| Optimization | AdamW LR 1e-4, 10-step linear warm-up, weight decay .01, gradient clip 1; microbatch 4, effective batch 8 (last batch smaller) |
| Seed | 20260918; deterministic CPU algorithms and shuffled order |

Evaluation, generation, downloads, and validation time are not included in training-loop timing. Benchmark updates were discarded; real training reloaded the original weights and reset the seed. This is one small LoRA treatment, not a hyperparameter search or full-SFT comparison. The research handoff recommended full SFT as the likely stronger treatment for an undertrained small model; LoRA was chosen here to honor the requested parameter-efficient experiment, not because superiority was established.

### Comparable held-out metrics

Exactly the same 63 records and 3,517 supervised assistant tokens, evaluated once per arm in fixed order with dropout disabled:

| Metric | Original | Candidate |
|---|---:|---:|
| Assistant-token mean NLL | 4.541893 | 4.471242 |
| Teacher-forced assistant-token accuracy | 25.419% | 26.187% |
| Manual task-content passes | 0 / 5 | 0 / 5 |

Labels include the existing multi-token `<|end|>` marker. Prompt and right-padding targets are ignored using `-1`, matching this GPT's loss function. Input filler remains token 0; it is never a target and cannot influence earlier positions under causal attention. No new tokenizer tokens or embedding resize were introduced. NLL and teacher-forced token accuracy are not free-generation correctness. **Do not compare these values with the legacy 2.5181 loss**, which counted padding and prompt tokens on a different evaluation procedure/data.

Complete answers are preserved, including the terminator. The encoder can remove old prompt tokens first, but this dataset rejects any example requiring that truncation. Selected sequences are at most 192 input tokens and answers at most 112 tokens; the model context remains 256. Long answers are rejected rather than cut.

### Raw before/after replies and content assessment

Identical formatting and greedy argmax decoding, at most 80 new tokens, stop on a completed `<|end|>` or `<|user|>` marker. Raw completions below retain leading whitespace, generated markers, repetition and malformed code. These prompts were fixed before training and excluded from the new data, with extra ML/subtraction/reverse-string/how-are-you filters on questions and answers. **Historical pretraining/old SFT membership is unknown**; unseen here means held out from this new experiment, not certified unseen throughout model history.

*(See results.json for full before/after reply comparisons)*

### Provenance, licenses and limitations

- **Databricks Dolly 15k**, Databricks employee-written instructions, [dataset and card](https://huggingface.co/datasets/databricks/databricks-dolly-15k), revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`: **[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)**. Some contexts derive from Wikipedia. Selected/normalized examples and citation-marker removal are modifications; retain attribution, license link and applicable share-alike obligations when distributing adapted data. No endorsement is implied.
- **OpenAssistant OASST1**, LAION/OpenAssistant contributors, [dataset and card](https://huggingface.co/datasets/OpenAssistant/oasst1), revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`: **Apache-2.0**, verified against pinned repository metadata, with full source LICENSE included. Used English, non-deleted, non-synthetic, root-level question/answer pairs, rank-zero answers, quality >= .5, toxicity <= .2, and zero flagged spam/task-failure/PII/inappropriate/hate labels. No ancestor context was dropped.
- Dolly categories favor QA, classification, extraction and brainstorming. Both sources are filtered for length, malformed code fences, persona boilerplate, special-marker injection, empty pairs and duplicates. English filtering for Dolly uses its documented language plus an ASCII-ratio heuristic, not a reliable language detector. Annotation/heuristic filters do **not** guarantee factual accuracy, safety, absence of private information, or flawless code. Only a sample was manually reviewed, not every answer. Coding coverage is sparse: 12 training prompts match python/function/code keywords, some of which are not programming tasks.
- Neither the dataset sample nor five diagnostic prompts constitute a broad benchmark. This experiment supports only a small token-prediction improvement, not useful answering or guaranteed future capability.

### Artifacts, validation and reuse

All paths below are relative to the extracted bundle's `apna_ai_model/` root.

- `experiments/lora_sft_v1/model_lora_sft_experimental.pt`: full native-GPT merged candidate; **experimental, not serving**.
- `experiments/lora_sft_v1/adapter_resume.pt`: native LoRA A/B weights, optimizer moments/step, model/training configuration, RNG state, shuffled order and training history.
- `checkpoints/model_finetuned.pt`, `checkpoints/model.pt`, `data/tokenizer.json`: original weights/tokenizer for local reproduction.
- `sft_core.py`, `prepare_sft.py`: reusable corrected pipeline.
- `config.json`, `manifest.json`, `train.jsonl`, `validation.jsonl`, `results.json`: frozen settings, data and evidence.

Four sanity tests passed: shifted assistant-only masking/padding; full-answer retention/overlength rejection; padding-invariant loss; zero-init equivalence, frozen-base/nonzero LoRA gradients, merge equivalence and tied embedding preservation.

**Next own-model step (not executed):** use the corrected objective to compare a small, bounded full-parameter SFT run against this frozen baseline, with manually checked short conversational/math/Python examples and a fresh unseen task set.
