import os, torch, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from peft import PeftModel
from datasets import load_dataset
import evaluate, sacrebleu
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from tqdm import tqdm

matplotlib.use('Agg')
LLAMA_ID = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER_PATH = "./llama-70b-nepali-refined-v2"
MBART_ID = "facebook/mbart-large-50-many-to-many-mmt"
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

class ResearchEvaluator:
    def __init__(self):
        self.chrf = evaluate.load("chrf")
        self.comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        print("Loading Base Llama for Zero-Shot...")
        self.base_l = AutoModelForCausalLM.from_pretrained(LLAMA_ID, quantization_config=bnb, device_map="auto", token=HF_TOKEN)
        self.l_tok = AutoTokenizer.from_pretrained(LLAMA_ID, token=HF_TOKEN)
        
        print("Loading mBART...")
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, dtype=torch.float16, device_map="auto")
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)

    def gen_llama(self, prompt):
        inputs = self.l_tok(prompt, return_tensors="pt").to(self.base_l.device)
        out = self.base_l.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=self.l_tok.eos_token_id)
        return self.l_tok.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

def run():
    ev = ResearchEvaluator(); samples = 100
    df = load_dataset("openlanguagedata/flores_plus", split='devtest').to_pandas()
    srcs = df[df['iso_639_3'] == 'eng']['text'].tolist()[:samples]
    refs = df[df['iso_639_3'] == 'npi']['text'].tolist()[:samples]

    data = {"mBART": [], "Llama_ZS": [], "EchoRefine_FT": []}

    print("Evaluating Baselines...")
    for s in tqdm(srcs):
        # mBART
        self_n_tok = ev.n_tok; self_n_tok.src_lang = "en_XX"
        d = ev.n_mod.generate(**self_n_tok(s, return_tensors="pt").to("cuda"), forced_bos_token_id=self_n_tok.lang_code_to_id["ne_NP"])[0]
        mbart_res = self_n_tok.decode(d, skip_special_tokens=True)
        data["mBART"].append(mbart_res)
        
        # Zero Shot
        data["Llama_ZS"].append(ev.gen_llama(f"Translate to Nepali: {s}\nRESULT:"))

    print("Evaluating Fine-Tuned...")
    ev.base_l = PeftModel.from_pretrained(ev.base_l, ADAPTER_PATH)
    for i, s in enumerate(tqdm(srcs)):
        mbart_d = data["mBART"][i]
        b = ev.n_mod.generate(**ev.n_tok(mbart_d, return_tensors="pt").to("cuda"), forced_bos_token_id=ev.n_tok.lang_code_to_id["en_XX"])[0]
        back_en = ev.n_tok.decode(b, skip_special_tokens=True)
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSource: {s}\nDraft: {mbart_d}\nBack-trans: {back_en}\n\nRESULT:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        data["EchoRefine_FT"].append(ev.gen_llama(prompt))

    # Metrics
    final = {}
    for k in ["mBART", "Llama_ZS", "EchoRefine_FT"]:
        preds = data[k]
        b = sacrebleu.corpus_bleu(preds, [[r] for r in refs]).score
        c = ev.chrf.compute(predictions=preds, references=[[r] for r in refs])['score']
        cm = ev.comet.compute(predictions=preds, references=refs, sources=srcs)['mean_score'] * 100
        final[k] = {"BLEU": round(b, 2), "chrF": round(c, 2), "COMET": round(cm, 2)}

    with open("results.json", "w") as f: json.dump(final, f, indent=4)
    
    # Plotting
    metrics = ["BLEU", "chrF", "COMET"]
    df_plot = pd.DataFrame(final).T
    ax = df_plot.plot(kind='bar', figsize=(12, 6), color=['#34495e', '#3498db', '#27ae60'])
    plt.title("EchoRefine: Final Research Results"); plt.savefig("final_results.png")

if __name__ == "__main__": run()
