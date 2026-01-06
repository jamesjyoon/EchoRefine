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
QE_MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Parameters
NUM_CANDIDATES = 5  
TEMPERATURE = 0.6
PROGRESS_FILE = "evaluation_progress.jsonl" 

class ResumableEvaluator:
    def __init__(self):
        print(">>> Loading Metrics, Judges & Models...")
        self.chrf = evaluate.load("chrf")
        self.comet_ref = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        
        self.mbr_scorer = SentenceTransformer("all-MiniLM-L6-v2")
        
        os.environ["HF_TOKEN"] = HF_TOKEN
        qe_path = download_model(QE_MODEL_NAME)
        self.qe_judge = load_from_checkpoint(qe_path).to("cuda")
        
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        base = AutoModelForCausalLM.from_pretrained(
            LLAMA_ID, quantization_config=bnb, device_map="auto", token=HF_TOKEN
        )
        # Attach adapter
        self.base_l = PeftModel.from_pretrained(base, ADAPTER_PATH)
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID, token=HF_TOKEN)
        self.l_tok.pad_token = self.l_tok.eos_token
        
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, dtype=torch.float16, device_map="auto")

    def mbart_translate(self, text, src="en_XX", tgt="ne_NP"):
        self.n_tok.src_lang = src
        inputs = self.n_tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.n_mod.device)
        out = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt])
        return self.n_tok.decode(out[0], skip_special_tokens=True).strip()

    def generate_zero_shot(self, source):
        """Generates Zero-Shot translation (Adapter Temporarily Disabled)."""
        with self.base_l.disable_adapter():
            messages = [{"role": "user", "content": f"Translate English to Nepali: {source}"}]
            prompt = self.l_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.l_tok(prompt, return_tensors="pt").to(self.base_l.device)
            out = self.base_l.generate(
                **inputs, max_new_tokens=256, do_sample=False, pad_token_id=self.l_tok.eos_token_id
            )
            return self.l_tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    def generate_candidates(self, src, draft, back_en):
        # Prompt matching Training Data
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
            if "RESULT:" in res: res = res.split("RESULT:")[-1].strip()
            candidates.append(res.strip())
        return list(set(candidates))

    def hybrid_selection(self, source, mbart_draft, llm_candidates):
        if not llm_candidates: return mbart_draft, "mBART"

        # 1. MBR Selection
        if len(llm_candidates) > 1:
            embeddings = self.mbr_scorer.encode(llm_candidates, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(embeddings, embeddings).sum(dim=1)
            best_llm = llm_candidates[torch.argmax(cos_scores).item()]
        else:
            best_llm = llm_candidates[0]

        # 2. Gatekeeping (Strict Improvement)
        data = [{"src": source, "mt": mbart_draft}, {"src": source, "mt": best_llm}]
        with torch.no_grad():
            qe_scores = self.qe_judge.predict(data, batch_size=2, gpus=1, progress_bar=False).scores
        
        if qe_scores[1] > qe_scores[0]:
            return best_llm, "LLM"
        return mbart_draft, "mBART"

def run_resumable_benchmark():
    # 1. Load Data
    try:
        df = load_dataset("openlanguagedata/flores_plus", split='devtest', trust_remote_code=True).to_pandas()
    except:
        df = load_dataset("openlanguagedata/flores_plus", split='devtest', trust_remote_code=True, storage_options={'timeout': 600}).to_pandas()

    srcs = df[df['iso_639_3'] == 'eng']['text'].tolist()
    refs = df[df['iso_639_3'] == 'npi']['text'].tolist()
    
    # 2. Check for Progress
    results_so_far = []
    if os.path.exists(PROGRESS_FILE):
        print(f">>> Found progress file {PROGRESS_FILE}. Loading...")
        with open(PROGRESS_FILE, 'r') as f:
            for line in f:
                try:
                    results_so_far.append(json.loads(line))
                except json.JSONDecodeError:
                    continue # Skip corrupted lines
        print(f">>> Resuming from index {len(results_so_far)}/{len(srcs)}")
    
    start_idx = len(results_so_far)
    if start_idx >= len(srcs):
        print("Evaluation already complete!")
        finalize_results(results_so_far, srcs, refs)
        return

    # 3. Init Model (Only if work remains)
    ev = ResumableEvaluator()

    # 4. Processing Loop
    print(f">>> Processing remaining {len(srcs) - start_idx} samples...")
    
    with open(PROGRESS_FILE, 'a') as f:
        for i in tqdm(range(start_idx, len(srcs))):
            src = srcs[i]
            
            # A. Baselines
            draft = ev.mbart_translate(src)
            zero_shot = ev.generate_zero_shot(src)
            
            # B. EchoRefine
            back = ev.mbart_translate(draft, src="ne_NP", tgt="en_XX")
            candidates = ev.generate_candidates(src, draft, back)
            final, winner = ev.hybrid_selection(src, draft, candidates)
            
            # C. Save Row
            row = {
                "idx": i,
                "mBART": draft,
                "Llama_ZS": zero_shot,
                "EchoRefine": final,
                "Winner": winner
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            
            results_so_far.append(row)

    # 5. Final Calculations
    finalize_results(results_so_far, srcs, refs)

def finalize_results(results, srcs, refs):
    print(">>> Calculating Final Metrics...")
    chrf = evaluate.load("chrf")
    comet_ref = evaluate.load("comet", "Unbabel/wmt22-comet-da")
    
    mbart_preds = [r["mBART"] for r in results]
    zs_preds = [r["Llama_ZS"] for r in results]
    echo_preds = [r["EchoRefine"] for r in results]
    
    limit = len(results)
    refs_nested = [[r] for r in refs[:limit]]
    refs_flat = refs[:limit]
    srcs_flat = srcs[:limit]

    metrics = {}
    for name, preds in [("mBART", mbart_preds), ("Llama_ZS", zs_preds), ("EchoRefine", echo_preds)]:
        b = sacrebleu.corpus_bleu(preds, refs_nested).score
        c = chrf.compute(predictions=preds, references=refs_nested)['score']
        cm = comet_ref.compute(predictions=preds, references=refs_flat, sources=srcs_flat, batch_size=8)['mean_score'] * 100
        metrics[name] = {"BLEU": round(b, 2), "chrF": round(c, 2), "COMET": round(cm, 2)}

    print(json.dumps(metrics, indent=4))
    with open("final_metrics_ne.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Plot
    df_plot = pd.DataFrame(metrics).T
    df_plot.plot(kind='bar', figsize=(10, 6), color=['#34495e', '#e67e22', '#27ae60'])
    plt.title(f"Final Research Results (N={limit})")
    plt.ylabel("Score")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("research_chart_ne.png")
    print("Done!")

if __name__ == "__main__":
    run_resumable_benchmark()