# =============================================================================
# ApnaAI GPU Training Notebook (Google Colab)
# =============================================================================
# Run on Google Colab with free T4 GPU.
# Fine-tunes Llama 3.2 1B using LoRA for efficient training.
#
# Two approaches:
#   1. Unsloth + Llama 3.2 1B (recommended, fastest)
#   2. Standard HuggingFace + LoRA (fallback)
#
# Upload as .ipynb or paste cells into Colab.
# =============================================================================

# %% [markdown]
# # ApnaAI - GPU Fine-Tuning with LoRA
# Train your own instruction-following AI model on Google Colab (free T4 GPU).

# %% Cell 1: Install Dependencies
# !pip install -q unsloth transformers datasets peft accelerate bitsandbytes trl

# %% Cell 2: Load Model with Unsloth (2x faster than standard)
from unsloth import FastLanguageModel
import torch

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

print(f"Trainable params: {model.print_trainable_parameters()}")

# %% Cell 3: Load & Format Training Data
from datasets import load_dataset

dataset = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

def format_chat(example):
    convs = example.get("conversations", [])
    messages = []
    for turn in convs:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

NUM_EXAMPLES = 10000  # ~2-3 hours on T4

print(f"Collecting {NUM_EXAMPLES} instruction examples...")
examples = []
for i, sample in enumerate(dataset):
    if i >= NUM_EXAMPLES:
        break
    convs = sample.get("conversations", [])
    if len(convs) >= 2:
        formatted = format_chat(sample)
        examples.append(formatted)
    if (i + 1) % 2000 == 0:
        print(f"  collected {len(examples)}/{NUM_EXAMPLES}")

print(f"Collected {len(examples)} examples")

from datasets import Dataset
train_dataset = Dataset.from_list(examples)

# %% Cell 4: Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        save_steps=500,
        output_dir="apna_ai_outputs",
        optim="adamw_8bit",
        seed=42,
    ),
)

print("Starting training...")
trainer_stats = trainer.train()
print(f"Training complete! Loss: {trainer_stats.training_loss:.4f}")

# %% Cell 5: Save Model
model.save_pretrained("apna_ai_lora")
tokenizer.save_pretrained("apna_ai_lora")
print("LoRA weights saved to apna_ai_lora/")

model.save_pretrained_merged("apna_ai_merged", tokenizer, save_method="merged_16bit")
print("Full merged model saved to apna_ai_merged/")

# Optional: GGUF for llama.cpp / Ollama
# model.save_pretrained_gguf("apna_ai_gguf", tokenizer, quantization_method="q4_k_m")

# %% Cell 6: Test
FastLanguageModel.for_inference(model)

def ask(question):
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=256, temperature=0.7, top_k=50)
    return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)

test_questions = [
    "What is machine learning?",
    "Write a Python function to check if a number is prime.",
    "Explain the theory of relativity in simple terms.",
    "Write a short poem about technology.",
    "How do neural networks learn?",
]

print("=" * 60)
print("MODEL TEST RESULTS")
print("=" * 60)
for q in test_questions:
    print(f"\nQ: {q}")
    print(f"A: {ask(q)}")
    print("-" * 40)

# %% Cell 7: Download
# !zip -r apna_ai_model.zip apna_ai_merged/
# from google.colab import files
# files.download('apna_ai_model.zip')

# %% [markdown]
# ## Approach 2: Standard HuggingFace (if Unsloth fails)
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

lora_config = LoraConfig(
    r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Then use SFTTrainer as in Cell 4 above
"""

print("\nScript ready. Run cells in Google Colab with T4 GPU runtime.")
