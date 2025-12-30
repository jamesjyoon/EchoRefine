import os
import torch
import json
import numpy as np
import pandas as pd
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

# --- Cluster Setup ---
import matplotlib
matplotlib.use('Agg') 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
LLAMA_ID = "meta-llama/Llama-3.1-70B-Instruct"
ADAPTER_PATH = "./llama-70b-nepali-refined" # Must match training output dir
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Language settings (English to Nepali)
SRC_ISO = "eng"
TGT_ISO = "npi"
TGT_MBART = "ne_NP"
LANG_NAME = "Nepali"

# Use an absolute path to avoid PEFT validation errors
PROJECT_ROOT = "/storage/ice1/6/3/jyoon370/EchoRefine_Project/EchoRefine"
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "llama-70b-nepali-refined")
   
# -----------------------------
# 1. Load Models & Metrics
# -----------------------------
class EchoRefineEvaluator:
    def __init__(self):

        if not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
            print(f"CRITICAL: Adapter not found at {ADAPTER_PATH}. Did training finish?")
            return # or exit
            
        print(">>> Loading Metrics...")
        self.chrf = evaluate.load("chrf")
        self.comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        
        print(">>> Loading Base Llama-3.1-70B...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_ID, quantization_config=bnb, device_map="auto", token=HUGGING_FACE_HUB_TOKEN
        )
        
        print(f">>> Attaching Adapter from {ADAPTER_PATH}...")
        self.l_mod = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID)
        
        print(">>> Loading mBART-50...")
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(
            MBART_ID, torch_dtype=torch.float16, device_map="auto"
        )

    def mbart_translate(self, text, src_tag, tgt_tag):
        self.n_tok.src_lang = src_tag
        inputs = self.n_tok(text, return_tensors="pt", truncation=True).to(self.n_mod.device)
        outputs = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt_tag])
        return self.n_tok.decode(outputs[0], skip_special_tokens=True)

    def refine_ft(self, source, draft, back_trans):
        # Must match the prompt format used during fine-tuning
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Source: {source}\nDraft: {draft}\nBack-trans: {back_trans}\n\n"
            f"RESULT:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = self.l_tok(prompt, return_tensors="pt").to(self.l_mod.device)
        outputs = self.l_mod.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=self.l_tok.eos_token_id)
        return self.l_tok.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

# -----------------------------
# 2. Main Evaluation Logic
# -----------------------------
def run_evaluation(num_samples=50):
    evaluator = EchoRefineEvaluator()
    
    print(">>> Loading FLORES dataset...")
    dataset = load_dataset("openlanguagedata/flores_plus", split='devtest')
    df = dataset.to_pandas()
    src_texts = df[df['iso_639_3'] == SRC_ISO]['text'].tolist()[:num_samples]
    ref_texts = df[df['iso_639_3'] == TGT_ISO]['text'].tolist()[:num_samples]

    results = {"mBART": [], "EchoRefine_FT": []}

    print(f">>> Running inference on {num_samples} samples...")
    for i in tqdm(range(num_samples)):
        src = src_texts[i]
        
        # mBART Baseline
        draft = evaluator.mbart_translate(src, "en_XX", TGT_MBART)
        
        # EchoRefine Pipeline
        back = evaluator.mbart_translate(draft, TGT_MBART, "en_XX")
        refined = evaluator.refine_with_cot(src, draft, back) # Using the FT model here
        
        results["mBART"].append(draft)
        results["EchoRefine_FT"].append(refined)

    # -----------------------------
    # 3. Metric Calculation
    # -----------------------------
    print(">>> Calculating Final Scores...")
    final_metrics = {}
    refs_nested = [[r] for r in ref_texts]

    for key in ["mBART", "EchoRefine_FT"]:
        preds = results[key]
        
        # BLEU (sacrebleu)
        bleu_score = sacrebleu.corpus_bleu(preds, refs_nested).score
        
        # chrF
        chrf_score = evaluator.chrf.compute(predictions=preds, references=refs_nested)['score']
        
        # COMET
        comet_res = evaluator.comet.compute(predictions=preds, references=ref_texts, sources=src_texts)
        comet_score = comet_res['mean_score'] * 100
        
        final_metrics[key] = {
            "BLEU": round(bleu_score, 2),
            "chrF": round(chrf_score, 2),
            "COMET": round(comet_score, 2)
        }

    # -----------------------------
    # 4. Save JSON Results
    # -----------------------------
    with open("actual_results.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(">>> Actual results saved to actual_results.json")

    # -----------------------------
    # 5. Generate Bar Chart
    # -----------------------------
    labels = ["BLEU", "chrF", "COMET"]
    mbart_vals = [final_metrics["mBART"][m] for m in labels]
    echo_vals = [final_metrics["EchoRefine_FT"][m] for m in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, mbart_vals, width, label='mBART Baseline', color='#34495e')
    rects2 = ax.bar(x + width/2, echo_vals, width, label='EchoRefine (Fine-Tuned)', color='#27ae60')

    ax.set_ylabel('Scores')
    ax.set_title(f'Translation Comparison: English to {LANG_NAME} (N={num_samples})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Label with heights
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('final_evaluation_results.png')
    print(">>> Bar chart saved as final_evaluation_results.png")

if __name__ == "__main__":
    # Ensure you set num_samples to 1012 for the final paper run
    run_evaluation(num_samples=50)
