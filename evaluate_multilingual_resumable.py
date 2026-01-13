import os
import torch
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import argparse
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

# --- Configuration ---
LLAMA_ID = "meta-llama/Llama-3.3-70B-Instruct"
# Update this if your path changed
ADAPTER_REFINE_PATH = "/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/llama-70b-multilingual-refined"
ADAPTER_DIRECT_PATH = "/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/llama-70b-direct-ft"
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
QE_MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Parameters
NUM_CANDIDATES = 5  
TEMPERATURE = 0.6

# Language Configs (Added Tamil, Removed Amharic as discussed)
LANG_MAP = {
    "npi": {"mbart": "ne_NP", "name": "Nepali"},
    "ben": {"mbart": "bn_IN", "name": "Bengali"},
    "sin": {"mbart": "si_LK", "name": "Sinhala"},
    "mya": {"mbart": "my_MM", "name": "Burmese"},
    "kor": {"mbart": "ko_KR", "name": "Korean"},
    "tam": {"mbart": "ta_IN", "name": "Tamil"},
    "hin": {"mbart": "hi_IN", "name": "Hindi"},
    "fra": {"mbart": "fr_XX", "name": "French"}
}

class MultilingualEvaluator:
    def __init__(self):
        print(">>> Loading Metrics, Judges & Models...")
        self.chrf = evaluate.load("chrf")
        self.comet_ref = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        self.mbr_scorer = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load CometKiwi Judge
        os.environ["HF_TOKEN"] = HF_TOKEN
        qe_path = download_model(QE_MODEL_NAME)
        self.qe_judge = load_from_checkpoint(qe_path).to("cuda")
        
        # Load Llama-70B Base
        print(">>> Loading Llama-70B Base...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.base_l = AutoModelForCausalLM.from_pretrained(
            LLAMA_ID, quantization_config=bnb, device_map="auto", token=HF_TOKEN
        )
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID, token=HF_TOKEN)
        self.l_tok.pad_token = self.l_tok.eos_token
        
        # Load Adapters
        print(f">>> Loading Refinement Adapter: {ADAPTER_REFINE_PATH}")
        self.base_l = PeftModel.from_pretrained(self.base_l, ADAPTER_REFINE_PATH, adapter_name="refine")
        
        if os.path.exists(ADAPTER_DIRECT_PATH):
            print(f">>> Loading Direct Adapter: {ADAPTER_DIRECT_PATH}")
            self.base_l.load_adapter(ADAPTER_DIRECT_PATH, adapter_name="direct")
            self.has_direct = True
        else:
            print(">>> WARNING: Direct Adapter not found. Skipping.")
            self.has_direct = False
        
        # Load mBART
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, dtype=torch.float16, device_map="auto")

    def mbart_translate(self, text, src_code, tgt_code):
        self.n_tok.src_lang = src_code
        inputs = self.n_tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.n_mod.device)
        out = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt_code])
        return self.n_tok.decode(out[0], skip_special_tokens=True).strip()

    def generate_zero_shot(self, source, lang_name):
        with self.base_l.disable_adapter():
            messages = [{"role": "user", "content": f"Translate English to {lang_name}: {source}"}]
            prompt = self.l_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.l_tok(prompt, return_tensors="pt").to(self.base_l.device)
            out = self.base_l.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=self.l_tok.eos_token_id)
            return self.l_tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    def generate_direct_ft(self, source, lang_name):
        if not self.has_direct: return "N/A"
        self.base_l.set_adapter("direct")
        messages = [
            {"role": "system", "content": f"You are a professional {lang_name} translator. Translate the English text accurately."},
            {"role": "user", "content": f"English: {source}\n{lang_name}:"}
        ]
        prompt = self.l_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.l_tok(prompt, return_tensors="pt").to(self.base_l.device)
        out = self.base_l.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=self.l_tok.eos_token_id)
        return self.l_tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    def generate_candidates_refine(self, src, draft, back_en, lang_name):
        self.base_l.set_adapter("refine")
        sys_msg = f"You are a professional {lang_name} editor. Fix the draft based on back-translation."
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Source: {src}\nDraft: {draft}\nBack-trans: {back_en}\n\n"
            f"Instruction: Fix the Draft based on the Back-trans. Keep the translation literal and consistent with the draft where correct.<|eot_id|>"
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
            if "RESULT:" in res: res = res.split("RESULT:")[-1].strip()
            candidates.append(res.strip())
        return list(set(candidates))

    def hybrid_selection(self, source, mbart_draft, llm_candidates):
        # 1. MBR Selection (This gives us the "Raw" LLM winner)
        if not llm_candidates: 
            best_llm = mbart_draft
        elif len(llm_candidates) > 1:
            embeddings = self.mbr_scorer.encode(llm_candidates, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(embeddings, embeddings).sum(dim=1)
            best_llm = llm_candidates[torch.argmax(cos_scores).item()]
        else:
            best_llm = llm_candidates[0]

        # 2. Gatekeeping (Compares LLM Winner vs mBART)
        data = [{"src": source, "mt": mbart_draft}, {"src": source, "mt": best_llm}]
        with torch.no_grad():
            qe_scores = self.qe_judge.predict(data, batch_size=2, gpus=1, progress_bar=False).scores
        
        # Strict Logic: Keep mBART unless LLM is strictly better
        final_output = best_llm if qe_scores[1] > qe_scores[0] else mbart_draft
        
        # Return BOTH the Raw LLM choice and the Gatekept Final choice
        return best_llm, final_output, ("LLM" if qe_scores[1] > qe_scores[0] else "mBART")

def run_benchmark(lang_iso, limit=None):
    if lang_iso not in LANG_MAP: raise ValueError(f"Language {lang_iso} not supported.")
    
    LANG_NAME = LANG_MAP[lang_iso]["name"]
    MBART_CODE = LANG_MAP[lang_iso]["mbart"]
    PROGRESS_FILE = f"progress_{lang_iso}.jsonl"
    
    print(f"\n>>> Starting Benchmark: {LANG_NAME} ({lang_iso})")

    # Load Data
    try:
        df = load_dataset("openlanguagedata/flores_plus", split='devtest', trust_remote_code=True).to_pandas()
    except:
        df = load_dataset("openlanguagedata/flores_plus", split='devtest', trust_remote_code=True, storage_options={'timeout': 600}).to_pandas()

    srcs = df[df['iso_639_3'] == 'eng']['text'].tolist()
    refs = df[df['iso_639_3'] == lang_iso]['text'].tolist()

    if limit: # <--- Apply logic
        srcs = srcs[:limit]
        refs = refs[:limit]
        
    # Resume Logic
    results_so_far = []
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            for line in f:
                try: results_so_far.append(json.loads(line))
                except: continue
        print(f">>> Resuming from index {len(results_so_far)}/{len(srcs)}")
    
    start_idx = len(results_so_far)
    if start_idx >= len(srcs):
        print("Evaluation complete!")
        finalize_results(results_so_far, srcs, refs, lang_iso)
        return

    ev = MultilingualEvaluator()

    with open(PROGRESS_FILE, 'a') as f:
        for i in tqdm(range(start_idx, len(srcs))):
            src = srcs[i]
            
            # A. Baselines
            draft = ev.mbart_translate(src, "en_XX", MBART_CODE)
            zero_shot = ev.generate_zero_shot(src, LANG_NAME)
            direct_ft = ev.generate_direct_ft(src, LANG_NAME)
            
            # B. EchoRefine
            back = ev.mbart_translate(draft, MBART_CODE, "en_XX")
            candidates = ev.generate_candidates_refine(src, draft, back, LANG_NAME)
            
            # Get RAW LLM winner AND Hybrid Final Result
            raw_llm, final_hybrid, winner = ev.hybrid_selection(src, draft, candidates)
            
            row = {
                "idx": i,
                "mBART": draft,
                "Llama_ZS": zero_shot,
                "Llama_Direct": direct_ft,
                "EchoRefine_Raw": raw_llm,      # <-- New Column: Ablation Study
                "EchoRefine_Hybrid": final_hybrid, # <-- Our Main Method
                "Winner": winner
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            results_so_far.append(row)

    finalize_results(results_so_far, srcs, refs, lang_iso)

def finalize_results(results, srcs, refs, lang_iso):
    print(">>> Calculating Metrics...")
    chrf = evaluate.load("chrf")
    comet_ref = evaluate.load("comet", "Unbabel/wmt22-comet-da")
    
    limit = len(results)
    refs_nested = [[r] for r in refs[:limit]]
    refs_flat = refs[:limit]
    srcs_flat = srcs[:limit]

    # Dynamically determine valid keys
    keys = ["mBART", "Llama_ZS", "EchoRefine_Raw", "EchoRefine_Hybrid"]
    if results[0]["Llama_Direct"] != "N/A": keys.insert(2, "Llama_Direct")

    metrics = {}
    for k in keys:
        preds = [r[k] for r in results]
        b = sacrebleu.corpus_bleu(preds, refs_nested).score
        c = chrf.compute(predictions=preds, references=refs_nested)['score']
        cm = comet_ref.compute(predictions=preds, references=refs_flat, sources=srcs_flat, batch_size=8)['mean_score'] * 100
        metrics[k] = {"BLEU": round(b, 2), "chrF": round(c, 2), "COMET": round(cm, 2)}

    print(json.dumps(metrics, indent=4))
    with open(f"results_{lang_iso}.json", "w") as f: json.dump(metrics, f, indent=4)
        
    df_plot = pd.DataFrame(metrics).T
    df_plot.plot(kind='bar', figsize=(12, 6))
    plt.title(f"Ablation Study: {lang_iso} (N={limit})")
    plt.ylabel("Score")
    plt.savefig(f"chart_{lang_iso}.png")
    print(f"Saved chart_{lang_iso}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()
    run_benchmark(args.lang, limit=args.limit)
