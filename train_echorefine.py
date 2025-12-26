import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, AutomodelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Config
MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"
OUTPUT_DIR = "./llama-70b-nepali-refined"
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
DATASET_ID = "opus100" # English-Nepali
NUM_SAMPLES = 5000     # Recommended size for QLoRA refinement
BATCH_SIZE = 16        # Speed up translation via batching

# 2. Load mBART-50
print("Loading mBART for synthetic data generation...")
tokenizer = AutoTokenizer.from_pretrained(MBART_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, torch_dtype=torch.float16).to("cuda")

# 3. Load English-Nepali Parallel Data
print("Loading OPUS-100 dataset...")
dataset = load_dataset(DATASET_ID, "en-ne", split="train", streaming=True)

def batch_translate(texts, src_lang, tgt_lang, tgt_token_id):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tgt_token_id,
            max_length=128
        )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

# 4. Processing Loop
data_rows = []
iterator = iter(dataset)
pbar = tqdm(total=NUM_SAMPLES)

ne_token_id = tokenizer.lang_code_to_id["ne_NP"]
en_token_id = tokenizer.lang_code_to_id["en_XX"]

current_batch_en = []
current_batch_target_ne = []

while len(data_rows) < NUM_SAMPLES:
    try:
        sample = next(iterator)
        current_batch_en.append(sample['translation']['en'])
        current_batch_target_ne.append(sample['translation']['ne'])
        
        if len(current_batch_en) == BATCH_SIZE:
            # A. Forward Translation (EN -> NE) to create the "Draft"
            drafts = batch_translate(current_batch_en, "en_XX", "ne_NP", ne_token_id)
            
            # B. Back-Translation (Draft NE -> EN) to create the "Context"
            back_trans = batch_translate(drafts, "ne_NP", "en_XX", en_token_id)
            
            # C. Store results
            for i in range(BATCH_SIZE):
                # Filter: Only keep if draft is actually different from the gold target
                # This ensures the model learns to "fix" rather than just "copy"
                if drafts[i].strip() != current_batch_target_ne[i].strip():
                    data_rows.append({
                        "source": current_batch_en[i],
                        "draft": drafts[i],
                        "back_trans": back_trans[i],
                        "target": current_batch_target_ne[i]
                    })
                    pbar.update(1)
                
                if len(data_rows) >= NUM_SAMPLES: break
                
            current_batch_en, current_batch_target_ne = [], []
            
    except StopIteration:
        break

# 5. Save to CSV
df = pd.DataFrame(data_rows)
df.to_csv("train_data.csv", index=False, encoding="utf-8")
print(f"\nSuccess! Generated {len(df)} samples for fine-tuning.")

# 6. Model Loading (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# 7. QLoRA Config
peft_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 8. Formatting for Llama-3
def formatting_func(example):
    return [
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a professional Nepali editor. Fix the draft based on back-translation.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Source: {src}\nDraft: {drft}\nBack-trans: {bt}\n\nRESULT:<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{tgt}<|eot_id|>"
        for src, drft, bt, tgt in zip(example['source'], example['draft'], example['back_trans'], example['target'])
    ]

# 9. Training
trainer = SFTTrainer(
    model=model,
    train_dataset=load_dataset("csv", data_files="train_data.csv", split="train"),
    peft_config=peft_config,
    formatting_func=formatting_func,
    max_seq_length=512,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=500, # Sufficient to see massive gains in low-res
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_8bit"
    ),
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
