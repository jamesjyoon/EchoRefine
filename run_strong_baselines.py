import argparse
import json
import os
from pathlib import Path

import evaluate
import sacrebleu
from datasets import load_dataset
from tqdm import tqdm

from baselines import Aya23Baseline, NLLB200Baseline


LANGS = ["npi", "ben", "sin", "mya", "kor", "tam", "hin", "fra"]
BASELINE_NAMES = {
    "nllb-200": "NLLB200",
    "aya-23": "Aya23",
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
DEFAULT_LOCAL_FLORES = Path("data/flores200/flores200_dataset/devtest")


def load_local_flores(lang_iso, flores_dir=DEFAULT_LOCAL_FLORES):
    flores_dir = Path(flores_dir)
    src_path = flores_dir / f"{FLORES_CODES['eng']}.devtest"
    ref_path = flores_dir / f"{FLORES_CODES[lang_iso]}.devtest"
    if not src_path.exists() or not ref_path.exists():
        return None
    return (
        src_path.read_text(encoding="utf-8").splitlines(),
        ref_path.read_text(encoding="utf-8").splitlines(),
    )


def load_flores(lang_iso, flores_dir=DEFAULT_LOCAL_FLORES):
    local = load_local_flores(lang_iso, flores_dir=flores_dir)
    if local:
        return local

    token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    dataset_kwargs = {"split": "devtest"}
    if token:
        dataset_kwargs["token"] = token
    try:
        dataset = load_dataset("openlanguagedata/flores_plus", **dataset_kwargs)
    except Exception:
        dataset_kwargs["storage_options"] = {"timeout": 600}
        dataset = load_dataset("openlanguagedata/flores_plus", **dataset_kwargs)
    df = dataset.to_pandas()
    sources = df[df["iso_639_3"] == "eng"]["text"].tolist()
    references = df[df["iso_639_3"] == lang_iso]["text"].tolist()
    return sources, references


def load_progress(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def compute_metrics(system_name, predictions, references, sources, skip_comet=False):
    refs_nested = [[ref] for ref in references]
    metrics = {
        "BLEU": round(sacrebleu.corpus_bleu(predictions, refs_nested).score, 2),
        "chrF": round(evaluate.load("chrf").compute(
            predictions=predictions,
            references=refs_nested,
        )["score"], 2),
    }
    if not skip_comet:
        comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        comet_inputs = {
            "predictions": predictions,
            "references": references,
            "sources": sources,
        }
        try:
            comet_score = comet.compute(**comet_inputs, batch_size=8)["mean_score"]
        except TypeError as exc:
            if "batch_size" not in str(exc):
                raise
            comet_score = comet.compute(**comet_inputs)["mean_score"]
        metrics["COMET"] = round(
            comet_score * 100,
            2,
        )
    return {system_name: metrics}


def build_baseline(args):
    token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if args.baseline == "nllb-200":
        return NLLB200Baseline(model_name=args.model_name or "facebook/nllb-200-distilled-600M")
    if args.baseline == "aya-23":
        return Aya23Baseline(
            model_name=args.model_name or "CohereLabs/aya-23-35B",
            token=token,
            load_in_4bit=not args.no_4bit,
        )
    raise ValueError(f"Unknown baseline: {args.baseline}")


def run_language(model, args, lang_iso):
    system_name = BASELINE_NAMES[args.baseline]
    progress_path = Path(args.progress_dir) / f"progress_{system_name.lower()}_{lang_iso}.jsonl"
    result_path = Path(args.output_dir) / f"results_{system_name.lower()}_{lang_iso}.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and result_path.exists():
        print(f"Skipping {system_name}-{lang_iso}: {result_path} exists")
        return

    sources, references = load_flores(lang_iso, flores_dir=args.flores_dir)
    if args.limit:
        sources = sources[:args.limit]
        references = references[:args.limit]

    rows = load_progress(progress_path)
    start_idx = len(rows)
    if start_idx < len(sources):
        with open(progress_path, "a", encoding="utf-8") as handle:
            for start in tqdm(
                range(start_idx, len(sources), args.batch_size),
                desc=f"{system_name}-{lang_iso}",
            ):
                batch_sources = sources[start:start + args.batch_size]
                translations = model.translate(
                    batch_sources,
                    src_lang="eng",
                    tgt_lang=lang_iso,
                    batch_size=args.batch_size,
                )
                for offset, translation in enumerate(translations):
                    row = {"idx": start + offset, system_name: translation}
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
                handle.flush()

    predictions = [row[system_name] for row in rows]
    used_references = references[:len(predictions)]
    used_sources = sources[:len(predictions)]
    metrics = compute_metrics(
        system_name,
        predictions,
        used_references,
        used_sources,
        skip_comet=args.skip_comet,
    )
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=4, ensure_ascii=False)
    print(json.dumps(metrics, indent=4, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Run stronger multilingual MT baselines.")
    parser.add_argument("--baseline", choices=["nllb-200", "aya-23"], required=True)
    parser.add_argument("--lang", default="all", help="Language ISO code or all.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--progress-dir", default=".")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--flores-dir", default=str(DEFAULT_LOCAL_FLORES))
    parser.add_argument("--skip-comet", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-4bit", action="store_true", help="Aya only: disable 4-bit loading.")
    args = parser.parse_args()

    langs = LANGS if args.lang == "all" else [args.lang]
    model = build_baseline(args)
    for lang_iso in langs:
        run_language(model, args, lang_iso)


if __name__ == "__main__":
    main()
