import os
import torch
import json
import numpy as np
import pandas as pd
import matplotlib
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
from comet import download_model, load_from_checkpoint

# --- Cluster Setup ---
matplotlib.use('Agg') 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Force offline if previously cached to avoid timeouts
# os.environ["HF_DATASETS_OFFLINE"] = "1" 

# Configuration
LLAMA_ID = "meta-llama/Llama-3.3-70B-Instruct"
# Ensure this matches your actual training output folder name
ADAPTER_PATH = "/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/llama-70b-nepali-refined-v2"
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
QE_MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

class ResearchEvaluator:
    def __init__(self):
        print(">>> Loading Metrics & Judges...")
        self.chrf = evaluate.load("chrf")
        self.comet_ref = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        
        # Internal Judge (QE)
        os.environ["HF_TOKEN"] = HF_TOKEN
        qe_path = download_model(QE_MODEL_NAME)
        self.qe_judge = load_from_checkpoint(qe_path).to("cuda")
        
        print(">>> Loading Models...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
        )
        
        # Load Base Llama
        self.base_l = AutoModelForCausalLM.from_pretrained(
            LLAMA_ID, quantization_config=bnb, device_map="auto", token=HF_TOKEN
        )
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID, token=HF_TOKEN)
        self.l_tok.pad_token = self.l_tok.eos_token
        
        # Load mBART
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, dtype=torch.float16, device_map="auto")

    # --- Helper: Generic Llama Generator ---
    def _generate_llama_raw(self, prompt, temperature=0.1):
        inputs = self.l_tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.base_l.device)
        
        # Determine sampling strategy based on temperature
        do_sample = temperature > 0
        
        outputs = self.base_l.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=do_sample, 
            temperature=temperature if do_sample else None,
            top_p=0.9 if do_sample else None,
            pad_token_id=self.l_tok.eos_token_id
        )
        return self.l_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    # --- Task 1: mBART Translation ---
    def mbart_translate(self, text, src="en_XX", tgt="ne_NP"):
        self.n_tok.src_lang = src
        inputs = self.n_tok(text, return_tensors="pt", truncation=True, max_length=1024).to(self.n_mod.device)
        out = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt])
        return self.n_tok.decode(out[0], skip_special_tokens=True).strip()

    # --- Task 2: Llama Zero-Shot ---
    def llama_zero_shot(self, source):
        # Proper prompt for direct translation
        messages = [
            {"role": "user", "content": f"Translate the following English text into Nepali. Provide ONLY the Nepali translation.\n\nEnglish: {source}\nNepali:"}
        ]
        prompt = self.l_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._generate_llama_raw(prompt, temperature=0.1)

    # --- Task 3: Llama Refinement (EchoRefine) ---
    def llama_refine(self, src, draft, back_en):
        # Prompt matching training format
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Source: {src}\nDraft: {draft}\nBack-trans: {back_en}\n\n"
            f"RESULT:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return self._generate_llama_raw(prompt, temperature=0.1)

    def select_best(self, source, mbart_cand, llama_cand):
        """Internal Judge Selection."""
        data = [{"src": source, "mt": mbart_cand}, {"src": source, "mt": llama_cand}]
        with torch.no_grad():
            scores = self.qe_judge.predict(data, batch_size=2, gpus=1, progress_bar=False).scores
        
        # Selection Logic: Prefer Llama if it's better
        if scores[1] > scores[0]:
            return llama_cand, "LLM"
        return mbart_cand, "mBART"

def run_research_benchmark(num_samples=100):
    ev = ResearchEvaluator()
    
    # Load Data
    print(">>> Loading Data...")
    try:
        df = load_dataset("openlanguagedata/flores_plus", split='devtest', trust_remote_code=True).to_pandas()
    except:
        # Fallback for network issues
        print("Standard load failed, trying with extended timeout...")
        df = load_dataset("openlanguagedata/flores_plus", split='devtest', trust_remote_code=True, storage_options={'timeout': 600}).to_pandas()

    srcs = df[df['iso_639_3'] == 'eng']['text'].tolist()[:num_samples]
    refs = df[df['iso_639_3'] == 'npi']['text'].tolist()[:num_samples]

    data = {"mBART": [], "Llama_ZeroShot": [], "EchoRefine_FT": []}
    selection_counts = {"LLM": 0, "mBART": 0}

    # --- PHASE 1: Baselines (Before loading adapter) ---
    print(">>> Phase 1: Generating Baselines (mBART & Zero-Shot)...")
    for s in tqdm(srcs):
        # mBART
        data["mBART"].append(ev.mbart_translate(s))
        
        # Llama Zero-Shot (No Adapter yet)
        data["Llama_ZeroShot"].append(ev.llama_zero_shot(s))

    # --- PHASE 2: EchoRefine (Load Adapter) ---
    print(f">>> Phase 2: Loading Adapter from {ADAPTER_PATH}...")
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"Adapter not found at {ADAPTER_PATH}")
        
    ev.base_l = PeftModel.from_pretrained(ev.base_l, ADAPTER_PATH)
    
    print(">>> Running EchoRefine Pipeline...")
    for i, s in enumerate(tqdm(srcs)):
        draft = data["mBART"][i]
        back = ev.mbart_translate(draft, src="ne_NP", tgt="en_XX")
        
        # Refine
        refined = ev.llama_refine(s, draft, back)
        
        # Judge
        best_sent, winner = ev.select_best(s, draft, refined)
        
        data["EchoRefine_FT"].append(best_sent)
        if winner == "LLM": selection_counts["LLM"] += 1
        else: selection_counts["mBART"] += 1

    print(f"\nSelection Statistics: {selection_counts}")

    # --- PHASE 3: Metrics ---
    final_metrics = {}
    for k in data.keys():
        preds = data[k]
        b = sacrebleu.corpus_bleu(preds, [[r] for r in refs]).score
        c = ev.chrf.compute(predictions=preds, references=[[r] for r in refs])['score']
        cm = ev.comet_ref.compute(predictions=preds, references=refs, sources=srcs)['mean_score'] * 100
        final_metrics[k] = {"BLEU": round(b, 2), "chrF": round(c, 2), "COMET": round(cm, 2)}

    # Save
    with open("results_paper.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(json.dumps(final_metrics, indent=4))

    # Plot
    df_plot = pd.DataFrame(final_metrics).T
    ax = df_plot.plot(kind='bar', figsize=(10, 6), color=['#2c3e50', '#e67e22', '#27ae60'])
    plt.title(f"Translation Performance (N={num_samples})")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("final_results_chart.png")
    print("Chart saved to final_results_chart.png")

if __name__ == "__main__":
    run_research_benchmark(num_samples=100)
