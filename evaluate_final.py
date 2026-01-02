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
QE_MODEL_NAME = "Unbabel/wmt22-cometkiwi-da" # Our Internal Judge
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

class EchoRefineUltimateEvaluator:
    def __init__(self):
        # 1. Load Internal QE Judge
        print(">>> Loading Internal QE Judge...")
        os.environ["HF_TOKEN"] = HF_TOKEN
        qe_path = download_model(QE_MODEL_NAME)
        self.qe_judge = load_from_checkpoint(qe_path).to("cuda")

        # 2. Load Evaluation Metrics
        self.chrf = evaluate.load("chrf")
        self.comet_ref = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        
        # 3. Load Models
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        base_l = AutoModelForCausalLM.from_pretrained(LLAMA_ID, quantization_config=bnb, device_map="auto", token=HF_TOKEN)
        self.l_mod = PeftModel.from_pretrained(base_l, ADAPTER_PATH)
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID, token=HF_TOKEN)
        
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, dtype=torch.float16, device_map="auto")

    def mbart_translate(self, text, src="en_XX", tgt="ne_NP"):
        self.n_tok.src_lang = src
        inputs = self.n_tok(text, return_tensors="pt", truncation=True).to(self.n_mod.device)
        out = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt])
        return self.n_tok.decode(out[0], skip_special_tokens=True).strip()

    def llama_refine(self, src, draft, back_en):
    # Ensure this prompt matches the 'prep' function in the training script exactly
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Source: {src}\n"
        f"Draft: {draft}\n"
        f"Back-trans: {back_en}\n\n"
        f"Instruction: Fix the Draft based on the Back-trans. "
        f"Keep the translation literal and consistent with the draft where correct.<|eot_id|>" # <-- INSERT HERE
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = self.l_tok(prompt, return_tensors="pt").to(self.l_mod.device)
    out = self.l_mod.generate(**inputs, max_new_tokens=256, do_sample=False)
    return self.l_tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    def get_best_sentence(self, source, mbart_cand, llama_cand):
        """Sentence-level selection using QE Judge."""
        data = [{"src": source, "mt": mbart_cand}, {"src": source, "mt": llama_cand}]
        with torch.no_grad():
            scores = self.qe_judge.predict(data, batch_size=2, gpus=1, progress_bar=False).scores
        
        # Pick the one the judge likes better
        if scores[1] > scores[0]:
            return llama_cand, "LLM"
        return mbart_cand, "mBART"

def run_research_benchmark(num_samples=100):
    ev = EchoRefineUltimateEvaluator()
    df = load_dataset("openlanguagedata/flores_plus", split='devtest').to_pandas()
    srcs = df[df['iso_639_3'] == 'eng']['text'].tolist()[:num_samples]
    refs = df[df['iso_639_3'] == 'npi']['text'].tolist()[:num_samples]

    final_outputs = {"mBART": [], "EchoRefine_Final": []}
    counts = {"LLM_Winner": 0, "mBART_Winner": 0}

    print(f">>> Processing {num_samples} samples...")
    for i in tqdm(range(num_samples)):
        src = srcs[i]
        
        # 1. Draft & Back-translate
        draft = ev.mbart_translate(src)
        back = ev.mbart_translate(draft, src="ne_NP", tgt="en_XX")
        
        # 2. Llama Refinement
        refined = ev.llama_refine(src, draft, back)
        
        # 3. SENTENCE-LEVEL SELECTION (The Magic Step)
        best, winner = ev.get_best_sentence(src, draft, refined)
        
        final_outputs["mBART"].append(draft)
        final_outputs["EchoRefine_Final"].append(best)
        
        if winner == "LLM": counts["LLM_Winner"] += 1
        else: counts["mBART_Winner"] += 1

    print(f"\nSelection Rate: {counts}")

    # --- Compute Final Metrics ---
    metrics = {}
    for key in ["mBART", "EchoRefine_Final"]:
        preds = final_outputs[key]
        r_nested = [[r] for r in refs]
        b = sacrebleu.corpus_bleu(preds, r_nested).score
        c = ev.chrf.compute(predictions=preds, references=r_nested)['score']
        cm = ev.comet_ref.compute(predictions=preds, references=refs, sources=srcs)['mean_score'] * 100
        metrics[key] = {"BLEU": round(b, 2), "chrF": round(c, 2), "COMET": round(cm, 2)}

    # Save and Plot (same as before)
    with open("ultimate_results.json", "w") as f: json.dump(metrics, f, indent=4)
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    run_research_benchmark(num_samples=100)
