import os
import torch
import json
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from peft import PeftModel
from datasets import load_dataset
import evaluate
import sacrebleu
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    BitsAndBytesConfig
)
import matplotlib # Added base import
matplotlib.use('Agg') # Set backend before importing pyplot
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
LLAMA_ID = "meta-llama/Llama-3.3-70B-Instruct"
# Ensure this path is correct for your scratch storage
ADAPTER_PATH = "/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/llama-70b-nepali-refined-v2"
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"

# Load Token - Ensure this is in your .env or set in shell
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

class EchoRefineEvaluator:
    def __init__(self):
        print(">>> Initializing Metrics (chrF, COMET)...")
        self.chrf = evaluate.load("chrf")
        self.comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
        )
        
        print(f">>> Loading Base Model: {LLAMA_ID}")
        # Pass token explicitly to handle gated repo access
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID, token=HF_TOKEN)
        self.l_tok.pad_token = self.l_tok.eos_token
        self.base_l = AutoModelForCausalLM.from_pretrained(
            LLAMA_ID, quantization_config=bnb, device_map="auto", 
            token=HF_TOKEN, dtype=torch.float16
        )
        
        print(">>> Loading mBART-50 Baseline...")
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(
            MBART_ID, dtype=torch.float16, device_map="auto"
        )

    def mbart_translate(self, text, src_tag="en_XX", tgt_tag="ne_NP"):
        self.n_tok.src_lang = src_tag
        inputs = self.n_tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.n_mod.device)
        out = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt_tag])
        return self.n_tok.decode(out[0], skip_special_tokens=True).strip()

    def llama_generate(self, prompt):
        inputs = self.l_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.base_l.device)
        outputs = self.base_l.generate(
            **inputs, max_new_tokens=256, do_sample=False, 
            pad_token_id=self.l_tok.eos_token_id
        )
        return self.l_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

def run_evaluation(num_samples=100):
    ev = EchoRefineEvaluator()
    df = load_dataset("openlanguagedata/flores_plus", split='devtest').to_pandas()
    srcs = df[df['iso_639_3'] == 'eng']['text'].tolist()[:num_samples]
    refs = df[df['iso_639_3'] == 'npi']['text'].tolist()[:num_samples]

    storage = {"mBART": [], "Llama_ZS": [], "EchoRefine_FT": []}

    print(">>> Stage 1: Evaluating mBART and Llama Zero-Shot...")
    for s in tqdm(srcs):
        # 1. mBART
        mbart_d = ev.mbart_translate(s)
        storage["mBART"].append(mbart_d)
        # 2. Zero-Shot
        zs_prompt = f"Translate the following English text to Nepali. Output only the text: {s}\nRESULT:"
        storage["Llama_ZS"].append(ev.llama_generate(zs_prompt))

    print(">>> Stage 2: Evaluating Fine-Tuned Model...")
    # Attach adapter to existing model to save VRAM
    ev.base_l = PeftModel.from_pretrained(ev.base_l, ADAPTER_PATH)
    
    for i, s in enumerate(tqdm(srcs)):
        mb_draft = storage["mBART"][i]
        back_en = ev.mbart_translate(mb_draft, src_tag="ne_NP", tgt_tag="en_XX")
        
        # Prompt must exactly match training format
        ft_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Source: {s}\nDraft: {mb_draft}\nBack-trans: {back_en}\n\nRESULT:<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        storage["EchoRefine_FT"].append(ev.llama_generate(ft_prompt))

    # --- Scoring ---
    final_metrics = {}
    for k in ["mBART", "Llama_ZS", "EchoRefine_FT"]:
        preds = storage[k]
        r_nested = [[r] for r in refs]
        b = sacrebleu.corpus_bleu(preds, r_nested).score
        c = ev.chrf.compute(predictions=preds, references=r_nested)['score']
        cm = ev.comet.compute(predictions=preds, references=refs, sources=srcs)['mean_score'] * 100
        final_metrics[k] = {"BLEU": round(b, 2), "chrF": round(c, 2), "COMET": round(cm, 2)}

    # --- Output ---
    with open("actual_results.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
    
    # --- Plotting ---
    plot_df = pd.DataFrame(final_metrics).T
    ax = plot_df.plot(kind='bar', figsize=(12, 7), color=['#34495e', '#3498db', '#27ae60'])
    plt.title(f"Comparison: mBART vs Llama-ZS vs EchoRefine-FT (N={num_samples})")
    plt.ylabel("Score (0-100)")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("final_research_chart.png", dpi=300)
    print("Results saved: actual_results.json and final_research_chart.png")

if __name__ == "__main__":
    run_evaluation(num_samples=100)
