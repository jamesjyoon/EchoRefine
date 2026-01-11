import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- Config ---
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
OUTPUT_DIR = "./llama-70b-direct-ft"
TRAIN_PATH = "train_data_multilingual.csv"
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# --- Model Loading (4-bit) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
)

print(f"Loading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto", token=HF_TOKEN
)
model = prepare_model_for_kbit_training(model)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# --- LoRA Config (Rank 32 for complex multilingual mapping) ---
peft_config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# --- Dynamic Prompting Function ---
def formatting_prompts_func(example):
    output_texts = []
    # Loop through the batch
    for i in range(len(example['source'])):
        lang = example['language'][i] # e.g., "Nepali", "Korean"
        # SIMPLIFIED PROMPT (No Draft, No Back-Trans)
        text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a professional {lang} translator. Translate the English text accurately.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"English: {example['source'][i]}\n"
            f"{lang}:<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{example['target'][i]}<|eot_id|>"
        ))
        output_texts.append(text)
    return output_texts

# --- Load & Prep Data ---
train_dataset = load_dataset("csv", data_files=TRAIN_PATH, split="train")

# --- Trainer Setup ---
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_seq_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, # Effective Batch = 16
    max_steps=1000, 
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    dataset_text_field="text", # Placeholder, formatting_func overrides
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func, # Uses the dynamic prompt
    args=sft_config
)

print(">>> Starting Multilingual Fine-Tuning...")
trainer.train()
model.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
