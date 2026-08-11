import argparse
import json
import os
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"source", "draft", "back_trans", "target"}


def token_levenshtein(a_tokens, b_tokens):
    if not a_tokens:
        return len(b_tokens)
    if not b_tokens:
        return len(a_tokens)

    previous = list(range(len(b_tokens) + 1))
    for i, a_token in enumerate(a_tokens, start=1):
        current = [i]
        for j, b_token in enumerate(b_tokens, start=1):
            cost = 0 if a_token == b_token else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def normalized_edit_distance(draft, target):
    draft_tokens = str(draft).split()
    target_tokens = str(target).split()
    denom = max(len(draft_tokens), len(target_tokens), 1)
    return token_levenshtein(draft_tokens, target_tokens) / denom


def sequence_similarity(a_text, b_text):
    return SequenceMatcher(None, str(a_text), str(b_text)).ratio()


def classify_error(row, edit_distance):
    draft = str(row["draft"]).strip()
    target = str(row["target"]).strip()
    source = str(row["source"]).strip()

    if draft == target:
        return "exact_duplicate"
    if edit_distance <= 0.05:
        return "near_duplicate"
    if draft.lower() == source.lower():
        return "source_copy"

    draft_len = max(len(draft.split()), 1)
    target_len = max(len(target.split()), 1)
    length_ratio = draft_len / target_len
    if length_ratio < 0.7 or length_ratio > 1.3:
        return "length_shift"
    if edit_distance >= 0.5:
        return "major_rewrite"
    return "lexical_or_morphological_edit"


def load_flores_sources():
    from datasets import load_dataset

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
    return set(df[df["iso_639_3"] == "eng"]["text"].tolist())


def compute_comet_backtranslation_scores(df, batch_size=8):
    import evaluate

    comet = evaluate.load("comet", "Unbabel/wmt22-comet-da")
    result = comet.compute(
        predictions=df["back_trans"].astype(str).tolist(),
        references=df["source"].astype(str).tolist(),
        sources=df["draft"].astype(str).tolist(),
        batch_size=batch_size,
    )
    return result.get("scores", [])


def summarize_numeric(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)), 4) if arr.size > 1 else 0.0,
        "p05": round(float(np.percentile(arr, 5)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
    }


def analyze_synthetic_dataset(
    dataset_path,
    include_flores_overlap=False,
    include_comet=False,
    comet_batch_size=8,
):
    df = pd.read_csv(dataset_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "language" not in df.columns:
        df["language"] = "unknown"

    edit_distances = [
        normalized_edit_distance(row["draft"], row["target"])
        for _, row in df.iterrows()
    ]
    backtranslation_similarity = [
        sequence_similarity(row["source"], row["back_trans"])
        for _, row in df.iterrows()
    ]
    error_types = [
        classify_error(row, edit_distance)
        for (_, row), edit_distance in zip(df.iterrows(), edit_distances)
    ]

    source_lengths = df["source"].astype(str).str.split().str.len().tolist()
    target_lengths = df["target"].astype(str).str.split().str.len().tolist()

    stats = {
        "dataset_path": str(dataset_path),
        "num_examples": int(len(df)),
        "language_counts": {
            str(key): int(value)
            for key, value in Counter(df["language"].astype(str)).items()
        },
        "source_length_tokens": summarize_numeric(source_lengths),
        "target_length_tokens": summarize_numeric(target_lengths),
        "normalized_draft_target_edit_distance": summarize_numeric(edit_distances),
        "backtranslation_source_similarity": summarize_numeric(backtranslation_similarity),
        "error_type_counts": {
            str(key): int(value)
            for key, value in Counter(error_types).items()
        },
        "draft_target_exact_duplicates": int(sum(str(a).strip() == str(b).strip() for a, b in zip(df["draft"], df["target"]))),
        "draft_target_near_duplicates_ed_le_0p05": int(sum(value <= 0.05 for value in edit_distances)),
    }

    if include_flores_overlap:
        flores_sources = load_flores_sources()
        source_set = set(df["source"].astype(str).tolist())
        overlap = source_set & flores_sources
        stats["flores_source_overlap"] = {
            "count": len(overlap),
            "rate": round(len(overlap) / max(len(source_set), 1), 6),
        }

    if include_comet:
        comet_scores = compute_comet_backtranslation_scores(df, batch_size=comet_batch_size)
        if comet_scores:
            stats["backtranslation_comet"] = summarize_numeric(np.asarray(comet_scores) * 100)

    return stats


def write_markdown(stats, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Synthetic Dataset Diagnostics",
        "",
        f"Dataset: `{stats['dataset_path']}`",
        f"Examples: {stats['num_examples']}",
        "",
        "## Language Counts",
        "",
    ]
    for language, count in sorted(stats["language_counts"].items()):
        lines.append(f"- {language}: {count}")

    lines.extend([
        "",
        "## Edit Distance",
        "",
        json.dumps(stats["normalized_draft_target_edit_distance"], indent=2),
        "",
        "## Back-Translation Similarity",
        "",
        json.dumps(stats["backtranslation_source_similarity"], indent=2),
        "",
        "## Error Type Counts",
        "",
    ])
    for error_type, count in sorted(stats["error_type_counts"].items()):
        lines.append(f"- {error_type}: {count}")

    if "flores_source_overlap" in stats:
        lines.extend([
            "",
            "## FLORES Source Overlap",
            "",
            json.dumps(stats["flores_source_overlap"], indent=2),
        ])
    if "backtranslation_comet" in stats:
        lines.extend([
            "",
            "## Back-Translation COMET",
            "",
            json.dumps(stats["backtranslation_comet"], indent=2),
        ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Validate EchoRefine synthetic refinement data.")
    parser.add_argument("--dataset", default="train_data_multilingual.csv")
    parser.add_argument("--output-json", default="results/tables/synthetic_dataset_diagnostics.json")
    parser.add_argument("--output-md", default="results/tables/synthetic_dataset_diagnostics.md")
    parser.add_argument("--flores-overlap", action="store_true")
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--comet-batch-size", type=int, default=8)
    args = parser.parse_args()

    stats = analyze_synthetic_dataset(
        args.dataset,
        include_flores_overlap=args.flores_overlap,
        include_comet=args.comet,
        comet_batch_size=args.comet_batch_size,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(stats, Path(args.output_md))
    print(f"Wrote diagnostics to {output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
