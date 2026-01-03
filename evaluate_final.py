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
from comet import download_model, load_from_checkpoint

# --- Cluster Setup ---
import matplotlib
matplotlib.use('Agg') 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
LLAMA_ID = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER_PATH = "/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/llama-70b-nepali-refined-v2"
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
QE_MODEL_NAME = "Unbabel/wmt22-cometkiwi-da" # Internal Judge
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

class ResearchEvaluator:
    def __init__(self):
        print(">>> Loading Metrics & Judges...")
        self.chrf = evaluate.load("chrf")
        self.comet_ref = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        
        os.environ["HF_TOKEN"] = HF_TOKEN
        qe_path = download_model(QE_MODEL_NAME)
        self.qe_judge = load_from_checkpoint(qe_path).to("cuda")
        
        print(">>> Loading mBART...")
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, dtype=torch.float16, device_map="auto")

        print(">>> Loading Base Llama-3.3 (4-bit)...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.l_mod = AutoModelForCausalLM.from_pretrained(LLAMA_ID, quantization_config=bnb, device_map="auto", token=HF_TOKEN)
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID, token=HF_TOKEN)
        self.l_tok.pad_token = self.l_tok.eos_token # Ensure pad token is set

    def mbart_translate(self, text, src="en_XX", tgt="ne_NP"):
        self.n_tok.src_lang = src
        inputs = self.n_tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.n_mod.device)
        out = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt])
        return self.n_tok.decode(out[0], skip_special_tokens=True).strip()

    def llama_refine(self, src, draft, back_en):
        # CORRECTED PROMPT: Must match the training prompt exactly
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Source: {src}\nDraft: {draft}\nBack-trans: {back_en}\n\n"
            f"Instruction: Fix the Draft based on the Back-trans. "
            f"Keep the translation literal and consistent with the draft where correct.<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = self.l_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.l_mod.device)
        out = self.l_mod.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=True, 
            temperature=0.3, # Use temperature as in previous version
            pad_token_id=self.l_tok.eos_token_id
        )
        return self.l_tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    def get_best_sentence(self, source, cand_a, cand_b):
        data = [{"src": source, "mt": cand_a}, {"src": source, "mt": cand_b}]
        with torch.no_grad():
            scores = self.qe_judge.predict(data, batch_size=2, gpus=1, progress_bar=False).scores
        
        # Soft Selection: Prefer LLM (cand_b) if it's within 0.05 of mBART
        if scores[1] > (scores[0] - 0.05):
            return cand_b, "LLM"
        return cand_a, "mBART"

def run_research_benchmark(num_samples=100):
    ev = ResearchEvaluator()
    df = load_dataset("openlanguagedata/flores_plus", split='devtest').to_pandas()
    srcs = df[df['iso_639_3'] == 'eng']['text'].tolist()[:num_samples]
    refs = df[df['iso_639_3'] == 'npi']['text'].tolist()[:num_samples]

    final_outputs = {"mBART": [], "Llama_ZeroShot": [], "EchoRefine_FT": []}
    counts = {"LLM_Winner": 0, "mBART_Winner": 0}

    # --- Phase 1: mBART & Llama Zero-Shot (Base Model) ---
    print(f">>> Phase 1: Zero-Shot Evaluation (N={num_samples})...")
    for s in tqdm(srcs):
        # mBART
        final_outputs["mBART"].append(ev.mbart_translate(s))
        
        # Zero-Shot Llama (Direct Translation)
        zs_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Translate the following English text to Nepali. Output only the translation.\nEnglish: {s}\nNepali:<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        # Using a slightly higher temp for ZS to encourage diversity if needed
        final_outputs["Llama_ZeroShot"].append(ev.llama_refine(s, "", "", temp=0.7)) # Pass "" for draft/back_en
                                                                                  # The prompt overrides them
    # --- Phase 2: Attach Adapter & Run EchoRefine ---
    print(f">>> Phase 2: Loading Fine-Tuned Adapter & Running EchoRefine...")
    ev.l_mod = PeftModel.from_pretrained(ev.l_mod, ADAPTER_PATH)
    
    for i, s in enumerate(tqdm(srcs)):
        draft = final_outputs["mBART"][i]
        back = ev.mbart_translate(draft, src="ne_NP", tgt="en_XX")
        
        # Fine-Tuned Refinement
        refined = ev.llama_refine(s, draft, back) # Pass through the refined llama_refine method
        
        # Sentence-Level Selection
        best, winner = ev.get_best_sentence(s, draft, refined)
        final_outputs["EchoRefine_FT"].append(best)
        
        if winner == "LLM": counts["LLM_Winner"] += 1
        else: counts["mBART_Winner"] += 1

    print(f"\nSelection Statistics: {counts}")

    # --- Metrics & Output ---
    metrics = {}
    r_nested = [[r] for r in refs]
    
    for k in final_outputs.keys():
        preds = final_outputs[k]
        b = sacrebleu.corpus_bleu(preds, r_nested).score
        c = ev.chrf.compute(predictions=preds, references=r_nested)['score']
        cm = ev.comet_ref.compute(predictions=preds, references=refs, sources=srcs)['mean_score'] * 100
        metrics[k] = {"BLEU": round(b, 2), "chrF": round(c, 2), "COMET": round(cm, 2)}

    # Save JSON
    with open("results_comparison.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(json.dumps(metrics, indent=4))

    # Save Graph
    df_res = pd.DataFrame(metrics).T
    ax = df_res.plot(kind='bar', figsize=(12, 6), color=['#34495e', '#e67e22', '#27ae60'])
    plt.title(f"Performance Comparison: English -> Nepali (N={num_samples})")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("full_comparison.png")
    print("Saved 'full_comparison.png'")

if __name__ == "__main__":
    run_research_benchmark(num_samples=100)