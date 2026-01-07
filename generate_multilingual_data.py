import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration ---
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
SAMPLES_PER_LANG = 1500  # 8 langs * 1500 = 12k total
BATCH_SIZE = 32

# Language Config: (OPUS_Code, mBART_Code, Language_Name)
LANG_CONFIGS = [
    # --- Low Resource ---
    ("ne", "ne_NP", "Nepali"),
    ("si", "si_LK", "Sinhala"),
    ("my", "my_MM", "Burmese"),
    ("ta", "ta_IN", "Tamil"),   
    
    # --- Mid/High Resource ---
    ("bn", "bn_IN", "Bengali"),
    ("hi", "hi_IN", "Hindi"),
    ("ko", "ko_KR", "Korean"),
    ("fr", "fr_XX", "French") 
]

# --- Setup ---
print("Loading mBART for Synthetic Data Generation...")
tokenizer = AutoTokenizer.from_pretrained(MBART_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, dtype=torch.float16).to("cuda")

def batch_translate(texts, src_lang, tgt_lang, tgt_token_id):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
    with torch.no_grad():
        generated_tokens = model.generate(**inputs, forced_bos_token_id=tgt_token_id, max_length=128)
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

all_data = []

# --- Generation Loop ---
for opus_code, mbart_code, lang_name in LANG_CONFIGS:
    print(f"\n>>> Processing {lang_name} ({opus_code})...")
    
    # Load OPUS-100 for this pair
    try:
        ds = load_dataset("opus100", f"en-{opus_code}", split="train", streaming=True)
    except:
        # Fallback for languages where order might be reversed in OPUS repo
        ds = load_dataset("opus100", f"{opus_code}-en", split="train", streaming=True)

    iterator = iter(ds)
    collected = 0
    pbar = tqdm(total=SAMPLES_PER_LANG)
    
    ne_id = tokenizer.lang_code_to_id[mbart_code]
    en_id = tokenizer.lang_code_to_id["en_XX"]

    while collected < SAMPLES_PER_LANG:
        batch_en, batch_tgt = [], []
        
        # 1. Fetch Batch
        for _ in range(BATCH_SIZE):
            try:
                sample = next(iterator)
                # Handle inconsistent OPUS formatting
                if 'en' in sample['translation']:
                    en_text = sample['translation']['en']
                    tgt_text = sample['translation'][opus_code]
                else:
                    continue # Skip malformed
                
                batch_en.append(en_text)
                batch_tgt.append(tgt_text)
            except StopIteration: break
        
        if not batch_en: break

        # 2. Generate Drafts & Back-Translations
        drafts = batch_translate(batch_en, "en_XX", mbart_code, ne_id)
        back_trans = batch_translate(drafts, mbart_code, "en_XX", en_id)

        # 3. Filter & Store
        for i in range(len(drafts)):
            if collected >= SAMPLES_PER_LANG: break
            
            # Only keep if Draft != Target (Validation: is there something to fix?)
            if drafts[i].strip() != batch_tgt[i].strip():
                all_data.append({
                    "language": lang_name, # Vital for the system prompt
                    "source": batch_en[i],
                    "draft": drafts[i],
                    "back_trans": back_trans[i],
                    "target": batch_tgt[i]
                })
                collected += 1
                pbar.update(1)

# --- Save Combined Dataset ---
df = pd.DataFrame(all_data)
df.to_csv("train_data_multilingual.csv", index=False)
print(f"\nSaved {len(df)} samples to train_data_multilingual.csv")
