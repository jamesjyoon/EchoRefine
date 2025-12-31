import os
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# --- Cluster Setup ---
import matplotlib
matplotlib.use('Agg') 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration - Ensure these match your training script exactly
LLAMA_ID = "meta-llama/Llama-3.3-70B-Instruct" 
ADAPTER_PATH = "/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/llama-70b-nepali-refined"
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

SRC_ISO = "eng"
TGT_ISO = "npi"
TGT_MBART = "ne_NP"
LANG_NAME = "Nepali"

class EchoRefineEvaluator:
    def __init__(self):
        print(">>> Loading Metrics...")
        self.chrf = evaluate.load("chrf")
        self.comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        
        print(f">>> Loading Base Model: {LLAMA_ID}")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_ID, quantization_config=bnb, device_map="auto", 
            token=HUGGING_FACE_HUB_TOKEN, dtype=torch.float16
        )
        
        print(f">>> Attaching Adapter from {ADAPTER_PATH}...")
        self.l_mod = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID, token=HUGGING_FACE_HUB_TOKEN)
        
        print(">>> Loading mBART-50...")
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(
            MBART_ID, dtype=torch.float16, device_map="auto"
        )

    def mbart_translate(self, text, src_tag, tgt_tag):
        self.n_tok.src_lang = src_tag
        # Added truncation and max_length to fix the warning
        inputs = self.n_tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.n_mod.device)
        outputs = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt_tag])
        return self.n_tok.decode(outputs[0], skip_special_tokens=True).strip()

    # RENAMED method to match the call in run_evaluation
    def refine_with_cot(self, source, draft, back_trans):
        # This MUST match the prompt format used in train_echorefine.py
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Source: {source}\nDraft: {draft}\nBack-trans: {back_trans}\n\n"
            f"RESULT:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = self.l_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.l_mod.device)
        outputs = self.l_mod.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=False, 
            pad_token_id=self.l_tok.eos_token_id
        )
        return self.l_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

def run_evaluation(num_samples=50):
    evaluator = EchoRefineEvaluator()
    
    print(">>> Loading FLORES dataset...")
    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    src_texts = df[df['iso_639_3'] == SRC_ISO]['text'].tolist()[:num_samples]
    ref_texts = df[df['iso_639_3'] == TGT_ISO]['text'].tolist()[:num_samples]

    results = {"mBART": [], "EchoRefine_FT": []}

    print(f">>> Processing {num_samples} samples...")
    for i in tqdm(range(num_samples)):
        src = src_texts[i]
        
        # 1. Generate mBART Baseline
        draft = evaluator.mbart_translate(src, "en_XX", TGT_MBART)
        
        # 2. Run EchoRefine (Back-translate + FT Refinement)
        back = evaluator.mbart_translate(draft, TGT_MBART, "en_XX")
        # Now calls the correct method name
        refined = evaluator.refine_with_cot(src, draft, back) 
        
        results["mBART"].append(draft)
        results["EchoRefine_FT"].append(refined)

    # --- Metrics & Charts (Same as previous implementation) ---
    print(">>> Calculating Final Scores...")
    final_metrics = {}
    refs_nested = [[r] for r in ref_texts]

    for key in ["mBART", "EchoRefine_FT"]:
        preds = results[key]
        b = sacrebleu.corpus_bleu(preds, refs_nested).score
        c = evaluator.chrf.compute(predictions=preds, references=refs_nested)['score']
        cm = evaluator.comet.compute(predictions=preds, references=ref_texts, sources=src_texts)['mean_score'] * 100
        
        final_metrics[key] = {"BLEU": round(b, 2), "chrF": round(c, 2), "COMET": round(cm, 2)}

    # Save JSON
    with open("actual_results.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    # Plot
    labels = ["BLEU", "chrF", "COMET"]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, [final_metrics["mBART"][m] for m in labels], width, label='mBART', color='#34495e')
    ax.bar(x + width/2, [final_metrics["EchoRefine_FT"][m] for m in labels], width, label='EchoRefine (FT)', color='#27ae60')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(); plt.savefig('final_evaluation_results.png')
    
    print("\n" + "="*30)
    print("EVALUATION COMPLETE")
    print("="*30)

if __name__ == "__main__":
    run_evaluation(num_samples=50)
