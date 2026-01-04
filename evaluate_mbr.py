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
from comet import download_model, load_from_checkpoint

# --- Cluster Setup ---
matplotlib.use('Agg') 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Config
LLAMA_ID = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER_PATH = "/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/llama-70b-nepali-refined-v2"
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
QE_MODEL_ID = "Unbabel/wmt22-cometkiwi-da" 
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Parameters
NUM_SAMPLES = 100
NUM_CANDIDATES = 5 
TEMPERATURE = 0.6 

class HybridEvaluator:
    def __init__(self):
        print(">>> Loading Metrics, Judges & Models...")
        self.chrf = evaluate.load("chrf")
        self.comet_ref = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        
        # 1. Semantic Scorer for MBR
        self.mbr_scorer = SentenceTransformer("all-MiniLM-L6-v2")
        
        # 2. Quality Estimation Judge
        os.environ["HF_TOKEN"] = HF_TOKEN
        qe_path = download_model(QE_MODEL_ID)
        self.qe_judge = load_from_checkpoint(qe_path).to("cuda")
        
        # 3. Llama + Adapter
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        base = AutoModelForCausalLM.from_pretrained(
            LLAMA_ID, quantization_config=bnb, device_map="auto", token=HF_TOKEN
        )
        self.base_l = PeftModel.from_pretrained(base, ADAPTER_PATH)
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID, token=HF_TOKEN)
        self.l_tok.pad_token = self.l_tok.eos_token
        
        # 4. mBART
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, dtype=torch.float16, device_map="auto")

    def mbart_translate(self, text, src="en_XX", tgt="ne_NP"):
        self.n_tok.src_lang = src
        inputs = self.n_tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.n_mod.device)
        out = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt])
        return self.n_tok.decode(out[0], skip_special_tokens=True).strip()

    def generate_zero_shot(self, source):
        """Generates Zero-Shot translation (Adapter Disabled)."""
        with self.base_l.disable_adapter():
            messages = [{"role": "user", "content": f"Translate English to Nepali: {source}"}]
            prompt = self.l_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.l_tok(prompt, return_tensors="pt").to(self.base_l.device)
            out = self.base_l.generate(
                **inputs, max_new_tokens=256, do_sample=False, pad_token_id=self.l_tok.eos_token_id
            )
            return self.l_tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    def generate_candidates(self, src, draft, back_en):
        """
        Generates N diverse candidates using the Adapter.
        Prompt matches 'train_echorefine.py' EXACTLY (User role + Instruction).
        """
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Source: {src}\n"
            f"Draft: {draft}\n"
            f"Back-trans: {back_en}\n\n"
            f"Instruction: Fix the Draft based on the Back-trans. "
            f"Keep the translation literal and consistent with the draft where correct.<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = self.l_tok(prompt, return_tensors="pt").to(self.base_l.device)
        
        candidates = []
        for _ in range(NUM_CANDIDATES):
            out = self.base_l.generate(
                **inputs, max_new_tokens=150, do_sample=True, 
                temperature=TEMPERATURE, top_p=0.9, pad_token_id=self.l_tok.eos_token_id
            )
            res = self.l_tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            # Clean up if model hallucinates tokens
            if "RESULT:" in res: res = res.split("RESULT:")[-1].strip()
            candidates.append(res.strip())
        return list(set(candidates))

    def hybrid_selection(self, source, mbart_draft, llm_candidates):
        """Stage 1: MBR -> Stage 2: QE Gatekeeping"""
        if not llm_candidates: return mbart_draft, "mBART"

        # 1. MBR Selection
        if len(llm_candidates) > 1:
            embeddings = self.mbr_scorer.encode(llm_candidates, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(embeddings, embeddings).sum(dim=1)
            best_llm = llm_candidates[torch.argmax(cos_scores).item()]
        else:
            best_llm = llm_candidates[0]

        # 2. QE Gatekeeping (vs mBART)
        data = [{"src": source, "mt": mbart_draft}, {"src": source, "mt": best_llm}]
        with torch.no_grad():
            qe_scores = self.qe_judge.predict(data, batch_size=2, gpus=1, progress_bar=False).scores
        
        # Only accept LLM if it strictly improves over mBART
        if qe_scores[1] > qe_scores[0]:
            return best_llm, "LLM"
        return mbart_draft, "mBART"

def run_hybrid_benchmark(num_samples=100):
    ev = HybridEvaluator()
    
    print(">>> Loading Data...")
    try:
        df = load_dataset("openlanguagedata/flores_plus", split='devtest', trust_remote_code=True).to_pandas()
    except:
        df = load_dataset("openlanguagedata/flores_plus", split='devtest', trust_remote_code=True, storage_options={'timeout': 600}).to_pandas()

    srcs = df[df['iso_639_3'] == 'eng']['text'].tolist()[:num_samples]
    refs = df[df['iso_639_3'] == 'npi']['text'].tolist()[:num_samples]

    storage = {"mBART": [], "Llama_ZS": [], "EchoRefine_Hybrid": []}
    stats = {"mBART": 0, "LLM": 0}

    print(f">>> Running Evaluation (N={num_samples})...")
    
    for i in tqdm(range(num_samples)):
        src = srcs[i]
        
        # 1. mBART
        draft = ev.mbart_translate(src)
        
        # 2. Zero-Shot
        zs = ev.generate_zero_shot(src)
        
        # 3. EchoRefine
        back = ev.mbart_translate(draft, src="ne_NP", tgt="en_XX")
        candidates = ev.generate_candidates(src, draft, back)
        final, winner = ev.hybrid_selection(src, draft, candidates)
        
        storage["mBART"].append(draft)
        storage["Llama_ZS"].append(zs)
        storage["EchoRefine_Hybrid"].append(final)
        stats[winner] += 1

    print(f"\nSelection Stats: {stats}")

    # --- Metrics ---
    final_metrics = {}
    for k in storage.keys():
        preds = storage[k]
        b = sacrebleu.corpus_bleu(preds, [[r] for r in refs]).score
        c = ev.chrf.compute(predictions=preds, references=[[r] for r in refs])['score']
        cm = ev.comet_ref.compute(predictions=preds, references=refs, sources=srcs)['mean_score'] * 100
        final_metrics[k] = {"BLEU": round(b, 2), "chrF": round(c, 2), "COMET": round(cm, 2)}

    # Output
    print(json.dumps(final_metrics, indent=4))
    with open("results_hybrid_full.json", "w") as f: json.dump(final_metrics, f, indent=4)
    
    # Plot
    df_plot = pd.DataFrame(final_metrics).T
    ax = df_plot.plot(kind='bar', figsize=(12, 6), color=['#34495e', '#e67e22', '#27ae60'])
    plt.title(f"Performance Comparison (N={num_samples})")
    plt.ylabel("Score")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("hybrid_full_results.png")

if __name__ == "__main__":
    run_hybrid_benchmark(num_samples=100)