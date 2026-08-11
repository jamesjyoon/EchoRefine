import os
import torch
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import argparse
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
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
DEFAULT_LOCAL_FLORES = "data/flores200/flores200_dataset/devtest"

# Parameters
NUM_CANDIDATES = 5
TEMPERATURE = 0.6
BATCH_SIZE = 8
DEFAULT_DECODING_METHOD = "mbr"
DEFAULT_GATE_MODE = "qe"
DEFAULT_QE_MARGIN = 0.0
DEFAULT_USE_BACKTRANSLATION = True

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

FLORES_CODES = {
    "eng": "eng_Latn",
    "npi": "npi_Deva",
    "ben": "ben_Beng",
    "sin": "sin_Sinh",
    "mya": "mya_Mymr",
    "kor": "kor_Hang",
    "tam": "tam_Taml",
    "hin": "hin_Deva",
    "fra": "fra_Latn",
}

def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")

def parse_beam_width(decoding_method):
    if not decoding_method.startswith("beam_search_"):
        return None
    try:
        return int(decoding_method.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Invalid beam-search decoding method: {decoding_method}") from exc

def validate_decoding_method(decoding_method):
    valid = {"mbr", "greedy"}
    if decoding_method in valid or parse_beam_width(decoding_method):
        return decoding_method
    raise ValueError(
        "decoding_method must be one of: mbr, greedy, beam_search_<N> "
        "(for example beam_search_4)."
    )

def build_run_suffix(
    num_candidates,
    decoding_method,
    qe_margin,
    gate_mode,
    use_backtranslation=DEFAULT_USE_BACKTRANSLATION,
    run_name=None,
):
    if run_name:
        return _slug(run_name)
    is_default = (
        num_candidates == NUM_CANDIDATES
        and decoding_method == DEFAULT_DECODING_METHOD
        and qe_margin == DEFAULT_QE_MARGIN
        and gate_mode == DEFAULT_GATE_MODE
        and use_backtranslation == DEFAULT_USE_BACKTRANSLATION
    )
    if is_default:
        return ""
    margin = str(qe_margin).replace(".", "p")
    bt_label = "bt" if use_backtranslation else "no_bt"
    return _slug(f"{decoding_method}_k{num_candidates}_{gate_mode}_m{margin}_{bt_label}")

def load_local_flores_devtest(flores_dir=DEFAULT_LOCAL_FLORES):
    if not os.path.isdir(flores_dir):
        return None
    records = []
    for iso, flores_code in FLORES_CODES.items():
        path = os.path.join(flores_dir, f"{flores_code}.devtest")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            for text in handle.read().splitlines():
                records.append({"iso_639_3": iso, "text": text})
    return pd.DataFrame.from_records(records)


def load_flores_plus_devtest():
    local_df = load_local_flores_devtest()
    if local_df is not None:
        return local_df

    dataset_kwargs = {"split": "devtest"}
    if HF_TOKEN:
        dataset_kwargs["token"] = HF_TOKEN
    try:
        return load_dataset("openlanguagedata/flores_plus", **dataset_kwargs).to_pandas()
    except Exception:
        dataset_kwargs["storage_options"] = {"timeout": 600}
        return load_dataset("openlanguagedata/flores_plus", **dataset_kwargs).to_pandas()

class MultilingualEvaluator:
    def __init__(self, load_qe=True, load_direct=True):
        print(">>> Loading Metrics, Judges & Models...")
        self.chrf = evaluate.load("chrf")
        self.comet_ref = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        self.mbr_scorer = SentenceTransformer("all-MiniLM-L6-v2")

        self.qe_judge = None
        if load_qe:
            # Load CometKiwi only for QE-gated runs; decoding/component ablations do
            # not need it during generation.
            if HF_TOKEN:
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
        self.l_tok.padding_side = "left" # Required for batched generation

        # Load Adapters
        print(f">>> Loading Refinement Adapter: {ADAPTER_REFINE_PATH}")
        self.base_l = PeftModel.from_pretrained(self.base_l, ADAPTER_REFINE_PATH, adapter_name="refine")

        if load_direct and os.path.exists(ADAPTER_DIRECT_PATH):
            print(f">>> Loading Direct Adapter: {ADAPTER_DIRECT_PATH}")
            self.base_l.load_adapter(ADAPTER_DIRECT_PATH, adapter_name="direct")
            self.has_direct = True
        else:
            if load_direct:
                print(">>> WARNING: Direct Adapter not found. Skipping.")
            else:
                print(">>> Skipping Direct Adapter for ablation-only run.")
            self.has_direct = False

        # Load mBART
        self.n_tok = AutoTokenizer.from_pretrained(MBART_ID)
        self.n_mod = AutoModelForSeq2SeqLM.from_pretrained(MBART_ID, dtype=torch.float16, device_map="auto")

    def mbart_translate_batch(self, texts, src_code, tgt_code):
        self.n_tok.src_lang = src_code
        inputs = self.n_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.n_mod.device)
        out = self.n_mod.generate(**inputs, forced_bos_token_id=self.n_tok.lang_code_to_id[tgt_code])
        return self.n_tok.batch_decode(out, skip_special_tokens=True)

    def generate_zero_shot_batch(self, sources, lang_name):
        with self.base_l.disable_adapter():
            prompts = []
            for source in sources:
                messages = [{"role": "user", "content": f"Translate English to {lang_name}: {source}"}]
                prompts.append(self.l_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

            inputs = self.l_tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(self.base_l.device)
            out = self.base_l.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=self.l_tok.eos_token_id)

            decoded = self.l_tok.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            return [d.strip() for d in decoded]

    def generate_direct_ft_batch(self, sources, lang_name):
        if not self.has_direct: return ["N/A"] * len(sources)
        self.base_l.set_adapter("direct")

        prompts = []
        for source in sources:
            messages = [
                {"role": "system", "content": f"You are a professional {lang_name} translator. Translate the English text accurately."},
                {"role": "user", "content": f"English: {source}\n{lang_name}:"}
            ]
            prompts.append(self.l_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        inputs = self.l_tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(self.base_l.device)
        out = self.base_l.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=self.l_tok.eos_token_id)

        decoded = self.l_tok.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return [d.strip() for d in decoded]

    def generate_candidates_refine_batch(
        self,
        srcs,
        drafts,
        back_ens,
        lang_name,
        num_candidates=NUM_CANDIDATES,
        decoding_method=DEFAULT_DECODING_METHOD,
        temperature=TEMPERATURE,
        use_backtranslation=DEFAULT_USE_BACKTRANSLATION,
    ):
        self.base_l.set_adapter("refine")
        decoding_method = validate_decoding_method(decoding_method)
        prompts = []
        for src, draft, back_en in zip(srcs, drafts, back_ens):
            if use_backtranslation:
                sys_msg = f"You are a professional {lang_name} editor. Fix the draft based on back-translation."
                user_content = (
                    f"Source: {src}\nDraft: {draft}\nBack-trans: {back_en}\n\n"
                    "Instruction: Fix the Draft based on the Back-trans. "
                    "Keep the translation literal and consistent with the draft where correct."
                )
            else:
                sys_msg = f"You are a professional {lang_name} editor. Fix the draft using only the source sentence."
                user_content = (
                    f"Source: {src}\nDraft: {draft}\n\n"
                    "Instruction: Fix the Draft using only the Source. "
                    "Keep the translation literal and consistent with the draft where correct."
                )
            prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_content}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            prompts.append(prompt)

        inputs = self.l_tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(self.base_l.device)

        beam_width = parse_beam_width(decoding_method)
        if decoding_method == "greedy":
            generation_kwargs = {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
            }
            return_sequences = 1
        elif beam_width:
            return_sequences = min(max(1, num_candidates), beam_width)
            generation_kwargs = {
                "do_sample": False,
                "num_beams": beam_width,
                "num_return_sequences": return_sequences,
            }
        else:
            return_sequences = max(1, num_candidates)
            generation_kwargs = {
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.9,
                "num_return_sequences": return_sequences,
            }

        out = self.base_l.generate(
            **inputs,
            max_new_tokens=150,
            pad_token_id=self.l_tok.eos_token_id,
            **generation_kwargs,
        )

        decoded_all = self.l_tok.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)

        # Reshape: (batch_size * return_sequences) -> (batch_size, return_sequences)
        candidates_batch = []
        for i in range(len(srcs)):
            sample_candidates = decoded_all[i*return_sequences : (i+1)*return_sequences]
            cleaned_candidates = []
            for res in sample_candidates:
                if "RESULT:" in res: res = res.split("RESULT:")[-1].strip()
                cleaned_candidates.append(res.strip())
            candidates_batch.append(list(set(cleaned_candidates))) # Deduplicate here if desired

        return candidates_batch

    def hybrid_selection_batch(
        self,
        sources,
        mbart_drafts,
        llm_candidates_batch,
        qe_margin=DEFAULT_QE_MARGIN,
        gate_mode=DEFAULT_GATE_MODE,
    ):
        if gate_mode not in {"qe", "always_llm", "always_anchor"}:
            raise ValueError("gate_mode must be one of: qe, always_llm, always_anchor")

        qe_inputs = []

        # 1. MBR Selection per sample
        best_llms = []
        for idx, (source, mbart_draft, llm_candidates) in enumerate(zip(sources, mbart_drafts, llm_candidates_batch)):
            if not llm_candidates:
                best_llm = mbart_draft
            elif len(llm_candidates) > 1:
                embeddings = self.mbr_scorer.encode(llm_candidates, convert_to_tensor=True)
                cos_scores = util.pytorch_cos_sim(embeddings, embeddings).sum(dim=1)
                best_llm = llm_candidates[torch.argmax(cos_scores).item()]
            else:
                best_llm = llm_candidates[0]
            best_llms.append(best_llm)

            # Prepare QE inputs (mBART vs Local Best LLM)
            qe_inputs.append({"src": source, "mt": mbart_draft})
            qe_inputs.append({"src": source, "mt": best_llm})

        # 2. Batch QE
        if gate_mode == "qe":
            if self.qe_judge is None:
                raise RuntimeError("QE gate requested but CometKiwi was not loaded.")
            with torch.no_grad():
                if qe_inputs:
                    qe_scores = self.qe_judge.predict(qe_inputs, batch_size=len(qe_inputs), gpus=1, progress_bar=False).scores
                else:
                    qe_scores = []
        else:
            qe_scores = [None] * len(qe_inputs)

        # 3. Decision Logic
        results = []
        for i in range(len(sources)):
            mbart_score = None if qe_scores[2*i] is None else float(qe_scores[2*i])
            llm_score = None if qe_scores[2*i+1] is None else float(qe_scores[2*i+1])
            best_llm = best_llms[i]

            if gate_mode == "always_llm":
                final_output = best_llm
                winner = "LLM"
            elif gate_mode == "always_anchor":
                final_output = mbart_drafts[i]
                winner = "mBART"
            else:
                use_llm = (llm_score - mbart_score) > qe_margin
                final_output = best_llm if use_llm else mbart_drafts[i]
                winner = "LLM" if use_llm else "mBART"

            qe_delta = None if mbart_score is None else llm_score - mbart_score
            results.append((best_llm, final_output, winner, mbart_score, llm_score, qe_delta))

        return results

def run_benchmark(
    lang_iso,
    limit=None,
    num_candidates=NUM_CANDIDATES,
    decoding_method=DEFAULT_DECODING_METHOD,
    qe_margin=DEFAULT_QE_MARGIN,
    gate_mode=DEFAULT_GATE_MODE,
    batch_size=BATCH_SIZE,
    temperature=TEMPERATURE,
    use_backtranslation=DEFAULT_USE_BACKTRANSLATION,
    run_name=None,
    progress_file=None,
    run_auxiliary_systems=True,
):
    if lang_iso not in LANG_MAP: raise ValueError(f"Language {lang_iso} not supported.")
    decoding_method = validate_decoding_method(decoding_method)

    LANG_NAME = LANG_MAP[lang_iso]["name"]
    MBART_CODE = LANG_MAP[lang_iso]["mbart"]
    suffix = build_run_suffix(
        num_candidates,
        decoding_method,
        qe_margin,
        gate_mode,
        use_backtranslation=use_backtranslation,
        run_name=run_name,
    )
    output_stem = f"{lang_iso}_{suffix}" if suffix else lang_iso
    PROGRESS_FILE = progress_file or f"progress_{output_stem}.jsonl"

    print(f"\n>>> Starting Benchmark: {LANG_NAME} ({lang_iso})")
    print(
        ">>> Config: "
        f"decoding={decoding_method}, k={num_candidates}, "
        f"gate={gate_mode}, qe_margin={qe_margin}, batch_size={batch_size}, "
        f"backtranslation={use_backtranslation}"
    )

    # Load Data
    df = load_flores_plus_devtest()

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
        finalize_results(results_so_far, srcs, refs, lang_iso, output_stem=output_stem)
        return

    ev = MultilingualEvaluator(load_qe=(gate_mode == "qe"), load_direct=run_auxiliary_systems)

    with open(PROGRESS_FILE, 'a') as f:
        # Step batches
        for i in tqdm(range(start_idx, len(srcs), batch_size)):
            batch_indices = list(range(i, min(i + batch_size, len(srcs))))
            batch_srcs = [srcs[idx] for idx in batch_indices]

            # --- Regrouped Operations to minimize Adapter Swaps ---

            # 1. mBART Initial Translation (CPU or separate model, no adapter needed)
            drafts = ev.mbart_translate_batch(batch_srcs, "en_XX", MBART_CODE)

            # 2. Base Model Generations (Zero Shot) - Switch to Base (disable adapters)
            # Actually disable_adapter() is a context manager but we can also just unset it?
            # Or just use the context manager around the batch.
            # But we want to reuse the loaded model state effectively.
            # MultilingualEvaluator helper methods handle adapter switching internaly.
            # Let's optimize: call methods in order.

            if run_auxiliary_systems:
                zero_shots = ev.generate_zero_shot_batch(batch_srcs, LANG_NAME)
            else:
                zero_shots = ["N/A"] * len(batch_srcs)

            # 3. Direct Adapter Generations
            if run_auxiliary_systems:
                direct_fts = ev.generate_direct_ft_batch(batch_srcs, LANG_NAME)
            else:
                direct_fts = ["N/A"] * len(batch_srcs)

            # 4. Optional mBART Back Translation
            if use_backtranslation:
                backs = ev.mbart_translate_batch(drafts, MBART_CODE, "en_XX")
            else:
                backs = [""] * len(drafts)

            # 5. Refinement Adapter Generations
            candidates_batch = ev.generate_candidates_refine_batch(
                batch_srcs,
                drafts,
                backs,
                LANG_NAME,
                num_candidates=num_candidates,
                decoding_method=decoding_method,
                temperature=temperature,
                use_backtranslation=use_backtranslation,
            )

            # 6. Selection
            selection_results = ev.hybrid_selection_batch(
                batch_srcs,
                drafts,
                candidates_batch,
                qe_margin=qe_margin,
                gate_mode=gate_mode,
            )

            # Write results
            for j, idx in enumerate(batch_indices):
                raw_llm, final_hybrid, winner, mbart_qe, llm_qe, qe_delta = selection_results[j]
                row = {
                    "idx": idx,
                    "Source": batch_srcs[j],
                    "Reference": refs[idx],
                    "mBART": drafts[j],
                    "Llama_ZS": zero_shots[j],
                    "Llama_Direct": direct_fts[j],
                    "EchoRefine_Raw": raw_llm,
                    "EchoRefine_Hybrid": final_hybrid,
                    "Winner": winner,
                    "mBART_QE": mbart_qe,
                    "EchoRefine_Raw_QE": llm_qe,
                    "QE_Delta": qe_delta,
                    "Decoding_Method": decoding_method,
                    "Num_Candidates": num_candidates,
                    "Gate_Mode": gate_mode,
                    "QE_Margin": qe_margin,
                    "Use_Backtranslation": use_backtranslation
                }
                f.write(json.dumps(row) + "\n")
                results_so_far.append(row)
            f.flush()

    finalize_results(results_so_far, srcs, refs, lang_iso, output_stem=output_stem)

def finalize_results(results, srcs, refs, lang_iso, output_stem=None):
    print(">>> Calculating Metrics...")
    if not results:
        print("No results to finalize.")
        return
    chrf = evaluate.load("chrf")
    comet_ref = evaluate.load("comet", "Unbabel/wmt22-comet-da")
    output_stem = output_stem or lang_iso

    limit = len(results)
    refs_nested = [[r] for r in refs[:limit]]
    refs_flat = refs[:limit]
    srcs_flat = srcs[:limit]

    # Dynamically determine valid keys
    keys = [
        key
        for key in ["mBART", "Llama_ZS", "Llama_Direct", "EchoRefine_Raw", "EchoRefine_Hybrid"]
        if all(row.get(key) != "N/A" for row in results)
    ]

    metrics = {}
    for k in keys:
        preds = [r[k] for r in results]
        b = sacrebleu.corpus_bleu(preds, refs_nested).score
        c = chrf.compute(predictions=preds, references=refs_nested)['score']
        cm = comet_ref.compute(predictions=preds, references=refs_flat, sources=srcs_flat)['mean_score'] * 100
        metrics[k] = {"BLEU": round(b, 2), "chrF": round(c, 2), "COMET": round(cm, 2)}

    print(json.dumps(metrics, indent=4))
    with open(f"results_{output_stem}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    df_plot = pd.DataFrame(metrics).T
    df_plot.plot(kind='bar', figsize=(14, 8), rot=45)
    plt.title(f"Ablation Study: {lang_iso} (N={limit})")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(f"chart_{output_stem}.png")
    print(f"Saved chart_{output_stem}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--num-candidates", type=int, default=NUM_CANDIDATES)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument(
        "--decoding-method",
        type=str,
        default=DEFAULT_DECODING_METHOD,
        help="mbr, greedy, or beam_search_<N>.",
    )
    parser.add_argument(
        "--qe-margin",
        type=float,
        default=DEFAULT_QE_MARGIN,
        help="Minimum CometKiwi advantage required before accepting the LLM candidate.",
    )
    parser.add_argument(
        "--gate-mode",
        choices=["qe", "always_llm", "always_anchor"],
        default=DEFAULT_GATE_MODE,
        help="Use QE gate, force raw LLM, or force the mBART anchor.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--no-backtranslation",
        action="store_true",
        help="Omit the mBART back-translation signal from the refinement prompt.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--progress-file", type=str, default=None)
    parser.add_argument(
        "--skip-auxiliary-generations",
        action="store_true",
        help="Skip zero-shot and direct-adapter outputs when only ablation metrics are needed.",
    )
    args = parser.parse_args()
    run_benchmark(
        args.lang,
        limit=args.limit,
        num_candidates=args.num_candidates,
        decoding_method=args.decoding_method,
        qe_margin=args.qe_margin,
        gate_mode=args.gate_mode,
        batch_size=args.batch_size,
        temperature=args.temperature,
        use_backtranslation=not args.no_backtranslation,
        run_name=args.run_name,
        progress_file=args.progress_file,
        run_auxiliary_systems=not args.skip_auxiliary_generations,
    )
