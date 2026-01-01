import os, torch, gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
OUTPUT_DIR = "./llama-70b-nepali-refined-v2"
TRAIN_PATH = "train_data_10k.csv"
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto", token=HF_TOKEN)
model = prepare_model_for_kbit_training(model)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# High-Rank LoRA for complex linguistic mapping
peft_config = LoraConfig(
    r=32, # Increased from 16
    lora_alpha=64, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

def prep(example):
    return {"text": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSource: {example['source']}\nDraft: {example['draft']}\nBack-trans: {example['back_trans']}\n\nRESULT:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{example['target']}<|eot_id|>"}

train_ds = load_dataset("csv", data_files=TRAIN_PATH, split="train").map(prep)

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_seq_length=1024,
    dataset_text_field="text",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, # Larger effective batch size for stability
    max_steps=1000, 
    learning_rate=1e-4, # Slightly lower LR for higher rank
    fp16=True,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = SFTTrainer(model=model, train_dataset=train_ds, args=args)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
