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
from comet import download_model, load_from_checkpoint

# --- Setup ---
matplotlib.use('Agg') 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Config
LLAMA_ID = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER_PATH = "/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine/llama-70b-nepali-refined-v2"
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
QE_MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Parameters
NUM_CANDIDATES = 5  # Best-of-5
TEMPERATURE = 0.6   # Higher temp = more diversity in attempts

class BestOfNEvaluator:
    def __init__(self):
        print(">>> Loading Models...")
        self.chrf = evaluate.load("chrf")
        self.comet_ref = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        
        # Judge
        os.environ["HF_TOKEN"] = HF_TOKEN
        qe_path = download_model(QE_MODEL_NAME)
        self.qe_judge = load_from_checkpoint(qe_path).to("cuda")
        
        # Llama
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.base_l = AutoModelForCausalLM.from_pretrained(
            LLAMA_ID, quantization_config=bnb, device_map="auto", token=HF_TOKEN
        )
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID, token=HF_TOKEN)
        self.l_tok.pad_token = self.l_tok.eos_token
        
        # Attach Fine-Tuned Adapter
        self.base_l = PeftModel.from_pretrained(self.base_l, ADAPTER_PATH)
        
        # mBART
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, dtype=torch.float16, device_map="auto")

    def mbart_translate(self, text, src="en_XX", tgt="ne_NP"):
        self.n_tok.src_lang = src
        inputs = self.n_tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.n_mod.device)
        out = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt])
        return self.n_tok.decode(out[0], skip_special_tokens=True).strip()

    def generate_candidates(self, src, draft, back_en):
        """Generates N variations of the refinement."""
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Source: {src}\nDraft: {draft}\nBack-trans: {back_en}\n\n"
            f"RESULT:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = self.l_tok(prompt, return_tensors="pt").to(self.base_l.device)
        
        candidates = []
        # Generate N times with sampling
        for _ in range(NUM_CANDIDATES):
            outputs = self.base_l.generate(
                **inputs, 
                max_new_tokens=150, 
                do_sample=True,         # Enable sampling
                temperature=TEMPERATURE, 
                top_p=0.9,
                pad_token_id=self.l_tok.eos_token_id
            )
            decoded = self.l_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            # Clean artifacts
            if "RESULT:" in decoded: decoded = decoded.split("RESULT:")[-1]
            candidates.append(decoded.strip())
            
        return candidates

    def pick_winner(self, source, mbart_draft, candidates):
        """Scores mBART + 5 Llama options and picks the absolute best."""
        # Pool includes mBART draft + LLM candidates
        pool = [mbart_draft] + candidates
        
        # Prepare batch for CometKiwi
        data = [{"src": source, "mt": c} for c in pool]
        
        with torch.no_grad():
            scores = self.qe_judge.predict(data, batch_size=len(pool), gpus=1, progress_bar=False).scores
        
        best_idx = np.argmax(scores)
        best_text = pool[best_idx]
        
        # Log who won
        winner_type = "mBART" if best_idx == 0 else "LLM"
        return best_text, winner_type

def run_best_of_n(num_samples=100):
    ev = BestOfNEvaluator()
    print(">>> Loading Data...")
    df = load_dataset("openlanguagedata/flores_plus", split='devtest').to_pandas()
    srcs = df[df['iso_639_3'] == 'eng']['text'].tolist()[:num_samples]
    refs = df[df['iso_639_3'] == 'npi']['text'].tolist()[:num_samples]

    storage = {"mBART": [], "EchoRefine_BestOfN": []}
    stats = {"mBART": 0, "LLM": 0}

    print(f">>> Running Best-of-{NUM_CANDIDATES} Evaluation on {num_samples} samples...")
    
    for i in tqdm(range(num_samples)):
        src = srcs[i]
        
        # 1. Draft
        draft = ev.mbart_translate(src)
        back = ev.mbart_translate(draft, src="ne_NP", tgt="en_XX")
        
        # 2. Generate N Candidates
        candidates = ev.generate_candidates(src, draft, back)
        
        # 3. Pick the Winner
        final, winner = ev.pick_winner(src, draft, candidates)
        
        storage["mBART"].append(draft)
        storage["EchoRefine_BestOfN"].append(final)
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
    with open("best_of_n_results.json", "w") as f: json.dump(final_metrics, f, indent=4)

    # Plot
    df_plot = pd.DataFrame(final_metrics).T
    df_plot.plot(kind='bar', figsize=(10, 6), color=['#34495e', '#8e44ad', '#27ae60'])
    plt.title(f"Best-of-{NUM_CANDIDATES} Performance (N={num_samples})")
    plt.savefig("best_of_n_chart.png")

if __name__ == "__main__":
    run_best_of_n(num_samples=100)
