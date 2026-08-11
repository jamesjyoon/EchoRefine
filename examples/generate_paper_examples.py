import argparse
import json
import os
from pathlib import Path

import sacrebleu
from datasets import load_dataset


LANG_MAP = {
    "npi": "Nepali",
    "ben": "Bengali",
    "sin": "Sinhala",
    "mya": "Burmese",
    "kor": "Korean",
    "tam": "Tamil",
    "hin": "Hindi",
    "fra": "French",
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


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
        try:
            dataset = load_dataset("openlanguagedata/flores_plus", **dataset_kwargs)
        except Exception as retry_exc:
            raise RuntimeError(
                "Could not load gated dataset openlanguagedata/flores_plus. "
                "Set HUGGING_FACE_HUB_TOKEN/HF_TOKEN, or use progress files "
                "created by the updated evaluator with Source and Reference fields. "
                "As an offline fallback, download the original FLORES-200 archive "
                "to data/flores200/flores200_dataset/devtest."
            ) from retry_exc
    df = dataset.to_pandas()
    sources = df[df["iso_639_3"] == "eng"]["text"].tolist()
    references = df[df["iso_639_3"] == lang_iso]["text"].tolist()
    return sources, references


def sources_references_from_rows(rows):
    if all("Source" in row and "Reference" in row for row in rows):
        return (
            [row["Source"] for row in rows],
            [row["Reference"] for row in rows],
        )
    return None


def sentence_chrf(prediction, reference):
    return sacrebleu.CHRF().sentence_score(prediction, [reference]).score


def row_to_example(row, source, reference, category, note, scores):
    return {
        "idx": row["idx"],
        "category": category,
        "source": source,
        "reference": reference,
        "mBART": row.get("mBART", ""),
        "Llama_ZS": row.get("Llama_ZS", ""),
        "Llama_Direct": row.get("Llama_Direct", ""),
        "EchoRefine_Raw": row.get("EchoRefine_Raw", ""),
        "EchoRefine_Hybrid": row.get("EchoRefine_Hybrid", ""),
        "winner": row.get("Winner", ""),
        "scores": scores,
        "error_analysis": note,
    }


def select_examples_for_language(lang_iso, progress_path, min_delta=2.0, flores_dir=DEFAULT_LOCAL_FLORES):
    rows = load_jsonl(progress_path)
    row_sources_refs = sources_references_from_rows(rows)
    if row_sources_refs:
        sources, references = row_sources_refs
    else:
        sources, references = load_flores(lang_iso, flores_dir=flores_dir)

    candidates = {
        "accepted_refinement": [],
        "overcorrection_prevented": [],
        "anchor_bias_candidate": [],
    }

    for row_number, row in enumerate(rows):
        idx = int(row["idx"])
        local_idx = idx if len(references) > idx else row_number
        reference = references[local_idx]
        scores = {
            "mBART_chrF": sentence_chrf(row.get("mBART", ""), reference),
            "raw_chrF": sentence_chrf(row.get("EchoRefine_Raw", ""), reference),
            "hybrid_chrF": sentence_chrf(row.get("EchoRefine_Hybrid", ""), reference),
        }
        raw_delta = scores["raw_chrF"] - scores["mBART_chrF"]
        hybrid_delta = scores["hybrid_chrF"] - scores["mBART_chrF"]

        if row.get("Winner") == "LLM" and hybrid_delta >= min_delta:
            note = (
                "The QE gate accepted the MBR-selected refinement; sentence-level chrF "
                "suggests the accepted edit moved closer to the reference."
            )
            candidates["accepted_refinement"].append(
                (hybrid_delta, row_to_example(row, sources[local_idx], reference, "accepted_refinement", note, scores))
            )
        elif row.get("Winner") == "mBART" and raw_delta <= -min_delta:
            note = (
                "The raw refinement drifted away from the reference under sentence-level chrF, "
                "so the QE gate retained the anchor."
            )
            candidates["overcorrection_prevented"].append(
                (-raw_delta, row_to_example(row, sources[local_idx], reference, "overcorrection_prevented", note, scores))
            )
        elif row.get("Winner") == "mBART" and raw_delta >= min_delta:
            note = (
                "The raw refinement scored better than the anchor by sentence-level chrF, "
                "but the QE gate retained mBART; this is a candidate anchor-bias case for manual analysis."
            )
            candidates["anchor_bias_candidate"].append(
                (raw_delta, row_to_example(row, sources[local_idx], reference, "anchor_bias_candidate", note, scores))
            )

    selected = {}
    for category, ranked in candidates.items():
        ranked.sort(key=lambda item: item[0], reverse=True)
        if ranked:
            selected[category] = ranked[0][1]
    return selected


def write_markdown(examples, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# EchoRefine Qualitative Examples", ""]
    for lang_iso, lang_examples in examples.items():
        lines.extend([f"## {LANG_MAP.get(lang_iso, lang_iso)} ({lang_iso})", ""])
        for category, example in lang_examples.items():
            lines.extend([
                f"### {category.replace('_', ' ').title()}",
                "",
                f"- Source: {example['source']}",
                f"- mBART: {example['mBART']}",
                f"- EchoRefine Raw: {example['EchoRefine_Raw']}",
                f"- EchoRefine Hybrid: {example['EchoRefine_Hybrid']}",
                f"- Reference: {example['reference']}",
                f"- Winner: {example['winner']}",
                f"- Analysis: {example['error_analysis']}",
                "",
            ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative examples for the paper.")
    parser.add_argument("--lang", default="all", help="Language ISO code or all.")
    parser.add_argument("--progress-dir", default=".")
    parser.add_argument("--flores-dir", default=str(DEFAULT_LOCAL_FLORES))
    parser.add_argument("--min-delta", type=float, default=2.0)
    parser.add_argument("--output-json", default="results/examples/paper_examples.json")
    parser.add_argument("--output-md", default="results/examples/paper_examples.md")
    args = parser.parse_args()

    langs = LANG_MAP.keys() if args.lang == "all" else [args.lang]
    all_examples = {}
    for lang_iso in langs:
        progress_path = Path(args.progress_dir) / f"progress_{lang_iso}.jsonl"
        if not progress_path.exists():
            continue
        selected = select_examples_for_language(
            lang_iso,
            progress_path,
            min_delta=args.min_delta,
            flores_dir=args.flores_dir,
        )
        if selected:
            all_examples[lang_iso] = selected

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(all_examples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(all_examples, Path(args.output_md))
    print(f"Wrote examples for {len(all_examples)} languages.")


if __name__ == "__main__":
    main()
