import os
import torch
import pandas as pd
import gc
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, # Fixed typo here
    BitsAndBytesConfig, 
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import inspect

# 1. Config
MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"
OUTPUT_DIR = "./llama-70b-nepali-refined"
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
DATASET_ID = "opus100"
NUM_SAMPLES = 5000     
BATCH_SIZE = 16        
TRAIN_DATA_PATH = "train_data.csv"

# ---------------------------------------------------------
# 2. Step 1: Synthetic Data Prep (Conditional)
# ---------------------------------------------------------
# Only generate if the file doesn't exist to save time on resumes
if not os.path.exists(TRAIN_DATA_PATH):
    print(f"{TRAIN_DATA_PATH} not found. Generating synthetic data with mBART...")
    m_tokenizer = AutoTokenizer.from_pretrained(MBART_ID)
    m_model = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, torch_dtype=torch.float16).to("cuda")

    dataset = load_dataset(DATASET_ID, "en-ne", split="train", streaming=True)
    
    def batch_translate(texts, src_lang, tgt_lang, tgt_token_id):
        m_tokenizer.src_lang = src_lang
        inputs = m_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            generated_tokens = m_model.generate(**inputs, forced_bos_token_id=tgt_token_id, max_length=128)
        return m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    data_rows = []
    iterator = iter(dataset)
    ne_token_id = m_tokenizer.lang_code_to_id["ne_NP"]
    en_token_id = m_tokenizer.lang_code_to_id["en_XX"]
    
    pbar = tqdm(total=NUM_SAMPLES)
    while len(data_rows) < NUM_SAMPLES:
        batch_en, batch_tgt = [], []
        for _ in range(BATCH_SIZE):
            try:
                s = next(iterator)
                batch_en.append(s['translation']['en'])
                batch_tgt.append(s['translation']['ne'])
            except StopIteration: break
        
        if not batch_en: break
        drafts = batch_translate(batch_en, "en_XX", "ne_NP", ne_token_id)
        back_trans = batch_translate(drafts, "ne_NP", "en_XX", en_token_id)

        for i in range(len(drafts)):
            if drafts[i].strip() != batch_tgt[i].strip() and len(data_rows) < NUM_SAMPLES:
                data_rows.append({"source": batch_en[i], "draft": drafts[i], "back_trans": back_trans[i], "target": batch_tgt[i]})
                pbar.update(1)

    pd.DataFrame(data_rows).to_csv(TRAIN_DATA_PATH, index=False)
    
    # CRITICAL: Free up GPU memory for Llama-70B
    del m_model
    del m_tokenizer
    torch.cuda.empty_cache()
    gc.collect()
else:
    print(f"Found existing {TRAIN_DATA_PATH}. Skipping generation.")

# ---------------------------------------------------------
# 3. Step 2: Fine-Tuning Setup (Llama-3.1-70B)
# ---------------------------------------------------------

# Check for existing checkpoints to resume
last_checkpoint = None
if os.path.exists(OUTPUT_DIR) and len(os.listdir(OUTPUT_DIR)) > 0:
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint:
        print(f">>> Resuming training from checkpoint: {last_checkpoint}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
)

print(f"Loading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

def formatting_func(example):
    return [
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Source: {s}\nDraft: {d}\nBack-trans: {b}\n\nRESULT:<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{t}<|eot_id|>"
        for s, d, b, t in zip(example['source'], example['draft'], example['back_trans'], example['target'])
    ]

# 9. Training Configuration
# We start with the standard TrainingArguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=1000, 
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,            
    save_total_limit=2,        
    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False
)

# 10. Intelligent Trainer Initialization
# We check the SFTTrainer signature to see where it wants 'max_seq_length'
trainer_signature = inspect.signature(SFTTrainer.__init__)

trainer_kwargs = {
    "model": model,
    "train_dataset": load_dataset("csv", data_files=TRAIN_DATA_PATH, split="train"),
    "peft_config": peft_config,
    "formatting_func": formatting_func,
    "args": training_args,
}

# If the version of SFTTrainer expects max_seq_length as a direct argument:
if "max_seq_length" in trainer_signature.parameters:
    print(">>> Version detected: Passing max_seq_length to Trainer")
    trainer_kwargs["max_seq_length"] = 512
else:
    # If it's not in the Trainer, it might be in a newer Config object
    # or handled by the data collator. We'll set a default in the config if possible.
    print(">>> Version detected: max_seq_length not in Trainer signature. Attempting Config update.")
    try:
        from trl import SFTConfig
        sft_config = SFTConfig(
            max_seq_length=512,
            **training_args.to_dict()
        )
        trainer_kwargs["args"] = sft_config
    except:
        print(">>> Fallback: Proceeding without explicit max_seq_length.")

trainer = SFTTrainer(**trainer_kwargs)

# 11. Start Training
print(">>> Starting Fine-Tuning...")
trainer.train(resume_from_checkpoint=last_checkpoint)

# Final Save
model.save_pretrained(OUTPUT_DIR)
