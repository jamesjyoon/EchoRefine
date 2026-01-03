import os
import torch
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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
from sentence_transformers import SentenceTransformer, util

# --- Cluster Setup ---
matplotlib.use('Agg') 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Config
LLAMA_ID = "meta-llama/Llama-3.3-70B-Instruct"
# Check your path: make sure this points to your trained adapter
ADAPTER_PATH = "/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/llama-70b-nepali-refined-v2"
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# MBR Parameters
NUM_SAMPLES = 100   # Set to 1012 for final paper run
NUM_CANDIDATES = 5  # Generate 5 versions per sentence
TEMPERATURE = 0.6   # Diversity for MBR

class MBRFullEvaluator:
    def __init__(self):
        print(">>> Loading Metrics & Models...")
        self.chrf = evaluate.load("chrf")
        self.comet_ref = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        self.mbr_scorer = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load Llama-70B (Base)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        base = AutoModelForCausalLM.from_pretrained(
            LLAMA_ID, quantization_config=bnb, device_map="auto", token=HF_TOKEN
        )
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID, token=HF_TOKEN)
        self.l_tok.pad_token = self.l_tok.eos_token
        
        # Attach Fine-Tuned Adapter (We will toggle this on/off)
        print(">>> Attaching Adapter...")
        self.l_mod = PeftModel.from_pretrained(base, ADAPTER_PATH)
        
        # Load mBART
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, dtype=torch.float16, device_map="auto")

    def mbart_translate(self, text, src="en_XX", tgt="ne_NP"):
        self.n_tok.src_lang = src
        inputs = self.n_tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.n_mod.device)
        out = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt])
        return self.n_tok.decode(out[0], skip_special_tokens=True).strip()

    def generate_zero_shot(self, source):
        """Runs with Adapter DISABLED to get pure Llama-3.3 baseline."""
        # Context manager disables LoRA layers temporarily
        with self.l_mod.disable_adapter():
            messages = [{"role": "user", "content": f"Translate English to Nepali: {source}"}]
            prompt = self.l_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.l_tok(prompt, return_tensors="pt").to(self.l_mod.device)
            
            out = self.l_mod.generate(
                **inputs, max_new_tokens=256, do_sample=False, temperature=None, top_p=None,
                pad_token_id=self.l_tok.eos_token_id
            )
            return self.l_tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    def generate_mbr_candidates(self, src, draft, back_en):
        """Runs with Adapter ENABLED to get Fine-Tuned refinements."""
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Source: {src}\nDraft: {draft}\nBack-trans: {back_en}\n\n"
            f"RESULT:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = self.l_tok(prompt, return_tensors="pt").to(self.l_mod.device)
        
        candidates = []
        # Generate N variations
        for _ in range(NUM_CANDIDATES):
            out = self.l_mod.generate(
                **inputs, max_new_tokens=150, do_sample=True, 
                temperature=TEMPERATURE, top_p=0.9, pad_token_id=self.l_tok.eos_token_id
            )
            res = self.l_tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            if "RESULT:" in res: res = res.split("RESULT:")[-1]
            candidates.append(res.strip())
        
        # Include the original draft in the pool as an anchor
        candidates.append(draft)
        return list(set(candidates))

    def select_via_mbr(self, candidates):
        """Picks the candidate most semantically similar to the consensus."""
        if len(candidates) == 1: return candidates[0]
        
        embeddings = self.mbr_scorer.encode(candidates, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(embeddings, embeddings)
        scores = cos_scores.sum(dim=1) # Sum similarity to all others
        best_idx = torch.argmax(scores).item()
        
        return candidates[best_idx]

def run_full_benchmark():
    ev = MBRFullEvaluator()
    df = load_dataset("openlanguagedata/flores_plus", split='devtest').to_pandas()
    srcs = df[df['iso_639_3'] == 'eng']['text'].tolist()[:NUM_SAMPLES]
    refs = df[df['iso_639_3'] == 'npi']['text'].tolist()[:NUM_SAMPLES]

    data = {"mBART": [], "Llama_ZS": [], "EchoRefine_MBR": []}

    print(f">>> Running Full Evaluation (N={NUM_SAMPLES})...")
    
    for i in tqdm(range(NUM_SAMPLES)):
        src = srcs[i]
        
        # 1. mBART Baseline
        draft = ev.mbart_translate(src)
        back = ev.mbart_translate(draft, src="ne_NP", tgt="en_XX")
        data["mBART"].append(draft)
        
        # 2. Llama Zero-Shot (No Adapter)
        zs_trans = ev.generate_zero_shot(src)
        data["Llama_ZS"].append(zs_trans)
        
        # 3. EchoRefine MBR (With Adapter)
        candidates = ev.generate_mbr_candidates(src, draft, back)
        winner = ev.select_via_mbr(candidates)
        data["EchoRefine_MBR"].append(winner)

    # --- Metrics ---
    final_metrics = {}
    for k in data.keys():
        preds = data[k]
        b = sacrebleu.corpus_bleu(preds, [[r] for r in refs]).score
        c = ev.chrf.compute(predictions=preds, references=[[r] for r in refs])['score']
        cm = ev.comet_ref.compute(predictions=preds, references=refs, sources=srcs)['mean_score'] * 100
        final_metrics[k] = {"BLEU": round(b, 2), "chrF": round(c, 2), "COMET": round(cm, 2)}

    # Output
    print(json.dumps(final_metrics, indent=4))
    with open("results_mbr_full.json", "w") as f: json.dump(final_metrics, f, indent=4)

    # Plot
    df_plot = pd.DataFrame(final_metrics).T
    ax = df_plot.plot(kind='bar', figsize=(10, 6), color=['#2c3e50', '#e67e22', '#27ae60'])
    plt.title(f"Evaluation with MBR Decoding (N={NUM_SAMPLES})")
    plt.ylabel("Score")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("final_results_mbr.png")
    print("Saved final_results_mbr.png")

if __name__ == "__main__":
    run_full_benchmark()
