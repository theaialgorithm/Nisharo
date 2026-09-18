---
license: apache-2.0
dataset_info:
  features:
  - name: message_id
    dtype: string
  - name: parent_id
    dtype: string
  - name: user_id
    dtype: string
  - name: created_date
    dtype: string
  - name: text
    dtype: string
  - name: role
    dtype: string
  - name: lang
    dtype: string
  - name: review_count
    dtype: int32
  - name: review_result
    dtype: bool
  - name: deleted
    dtype: bool
  - name: rank
    dtype: int32
  - name: synthetic
    dtype: bool
  - name: model_name
    dtype: string
---

# OpenAssistant Conversations Dataset (OASST1)

## Dataset Description

- **Homepage:** https://www.open-assistant.io/
- **Repository:** https://github.com/LAION-AI/Open-Assistant
- **Paper:** https://arxiv.org/abs/2304.07327

### Dataset Summary

In an effort to democratize research on large-scale alignment, we release OpenAssistant
Conversations (OASST1), a human-generated, human-annotated assistant-style conversation
corpus consisting of 161,443 messages in 35 different languages, annotated with 461,292
quality ratings, resulting in over 10,000 fully annotated conversation trees. The corpus
is a product of a worldwide crowd-sourcing effort involving over 13,500 volunteers.

Please refer to our [paper](https://arxiv.org/abs/2304.07327) for further details.

### Dataset Structure

This dataset contains message trees. Each message tree has an initial prompt message as the root node,
which can have multiple child messages as replies, and these child messages can have multiple replies.

All messages have a role property: this can either be "assistant" or "prompter". The roles in
conversation threads from prompt to leaf node strictly alternate between "prompter" and "assistant".

This version of the dataset contains data collected on the [open-assistant.io](https://open-assistant.io/) website until April 12 2023.

## Main Dataset Files

Conversation data is provided either as nested messages in trees (extension `.trees.jsonl.gz`)
or as a flat list (table) of messages (extension `.messages.jsonl.gz`).

### Ready For Export Trees

```
2023-04-12_oasst_ready.trees.jsonl.gz       10,364 trees with 88,838 total messages
2023-04-12_oasst_ready.messages.jsonl.gz    88,838 messages
```

### Languages

OpenAssistant Conversations incorporates 35 different languages. Primary: English (71,956 messages).

## Contact

- Discord [Open Assistant Discord Server](https://ykilcher.com/open-assistant-discord)
- GitHub: [LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant)
- E-Mail: [open-assistant@laion.ai](mailto:open-assistant@laion.ai)
