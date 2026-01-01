import os, torch, pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
DATASET_ID = "opus100"
NUM_SAMPLES = 10000 # Doubled for better BLEU
BATCH_SIZE = 16

print("Loading mBART for data generation...")
tokenizer = AutoTokenizer.from_pretrained(MBART_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, torch_dtype=torch.float16).to("cuda")

dataset = load_dataset(DATASET_ID, "en-ne", split="train", streaming=True)
ne_id, en_id = tokenizer.lang_code_to_id["ne_NP"], tokenizer.lang_code_to_id["en_XX"]

def batch_translate(texts, src, tgt, tid):
    tokenizer.src_lang = src
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, forced_bos_token_id=tid, max_length=128)
    return tokenizer.batch_decode(out, skip_special_tokens=True)

data = []
iterator = iter(dataset)
pbar = tqdm(total=NUM_SAMPLES)

while len(data) < NUM_SAMPLES:
    batch_en, batch_tgt = [], []
    for _ in range(BATCH_SIZE):
        try:
            s = next(iterator); batch_en.append(s['translation']['en']); batch_tgt.append(s['translation']['ne'])
        except StopIteration: break
    
    if not batch_en: break
    drafts = batch_translate(batch_en, "en_XX", "ne_NP", ne_id)
    back = batch_translate(drafts, "ne_NP", "en_XX", en_id)

    for i in range(len(drafts)):
        # Only learn from sentences where the draft actually needs fixing
        if drafts[i].strip() != batch_tgt[i].strip() and len(data) < NUM_SAMPLES:
            data.append({"source": batch_en[i], "draft": drafts[i], "back_trans": back[i], "target": batch_tgt[i]})
            pbar.update(1)

pd.DataFrame(data).to_csv("train_data_10k.csv", index=False)
