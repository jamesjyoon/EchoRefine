import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
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

DEFAULT_SYSTEMS = [
    "mBART",
    "Llama_ZS",
    "Llama_Direct",
    "EchoRefine_Raw",
    "EchoRefine_Hybrid",
]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_local_flores_references(lang_iso, flores_dir=DEFAULT_LOCAL_FLORES):
    flores_dir = Path(flores_dir)
    src_path = flores_dir / f"{FLORES_CODES['eng']}.devtest"
    ref_path = flores_dir / f"{FLORES_CODES[lang_iso]}.devtest"
    if not src_path.exists() or not ref_path.exists():
        return None
    sources = src_path.read_text(encoding="utf-8").splitlines()
    references = ref_path.read_text(encoding="utf-8").splitlines()
    return sources, references


def load_flores_references(lang_iso, flores_dir=DEFAULT_LOCAL_FLORES):
    local = load_local_flores_references(lang_iso, flores_dir=flores_dir)
    if local:
        return local

    token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    dataset_kwargs = {"split": "devtest"}
    if token:
        dataset_kwargs["token"] = token
    try:
        dataset = load_dataset("openlanguagedata/flores_plus", **dataset_kwargs)
    except Exception as exc:
        dataset_kwargs["storage_options"] = {"timeout": 600}
        try:
            dataset = load_dataset("openlanguagedata/flores_plus", **dataset_kwargs)
        except Exception as retry_exc:
            raise RuntimeError(
                "Could not load gated dataset openlanguagedata/flores_plus. "
                "Set HUGGING_FACE_HUB_TOKEN/HF_TOKEN, or run analyses on progress "
                "files that include Source and Reference fields. As an offline fallback, "
                "download the original FLORES-200 archive to "
                "data/flores200/flores200_dataset/devtest."
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


def paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=1000, alpha=0.05, seed=13):
    """Return mean paired difference, CI, and bootstrap sign-test p-value."""
    diffs = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    if diffs.size == 0:
        raise ValueError("Cannot bootstrap an empty score list.")

    rng = np.random.default_rng(seed)
    sample_ids = rng.integers(0, diffs.size, size=(n_bootstrap, diffs.size))
    boot_means = diffs[sample_ids].mean(axis=1)
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    p_value = float(2 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0)))
    return {
        "mean_diff": float(diffs.mean()),
        "std_diff": float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": min(1.0, p_value),
    }


def significance_marker(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def sentence_metric_scores(metric_name, predictions, references, sources=None, batch_size=8):
    metric_name = metric_name.lower()
    if metric_name == "bleu":
        scorer = sacrebleu.BLEU(effective_order=True)
        return np.asarray(
            [scorer.sentence_score(pred, [ref]).score for pred, ref in zip(predictions, references)],
            dtype=float,
        )
    if metric_name == "chrf":
        scorer = sacrebleu.CHRF()
        return np.asarray(
            [scorer.sentence_score(pred, [ref]).score for pred, ref in zip(predictions, references)],
            dtype=float,
        )
    if metric_name == "comet":
        import evaluate as hf_evaluate

        if sources is None:
            raise ValueError("COMET scoring requires source sentences.")
        comet = hf_evaluate.load("comet", "Unbabel/wmt22-comet-da")
        result = comet.compute(
            predictions=predictions,
            references=references,
            sources=sources,
            batch_size=batch_size,
        )
        scores = result.get("scores")
        if scores is None:
            raise RuntimeError("COMET did not return per-segment scores.")
        return np.asarray(scores, dtype=float) * 100
    raise ValueError(f"Unsupported metric: {metric_name}")


def available_systems(rows):
    present = []
    for system in DEFAULT_SYSTEMS:
        if system in rows[0] and rows[0][system] != "N/A":
            present.append(system)
    return present


def analyze_language(
    lang_iso,
    progress_path,
    target_system,
    baseline_systems,
    metrics,
    n_bootstrap,
    seed,
    batch_size,
    flores_dir=DEFAULT_LOCAL_FLORES,
):
    rows = load_jsonl(progress_path)
    if not rows:
        raise ValueError(f"No rows found in {progress_path}")

    row_sources_refs = sources_references_from_rows(rows)
    if row_sources_refs:
        sources, references = row_sources_refs
    else:
        sources, references = load_flores_references(lang_iso, flores_dir=flores_dir)
        idxs = [int(row.get("idx", i)) for i, row in enumerate(rows)]
        sources = [sources[idx] for idx in idxs]
        references = [references[idx] for idx in idxs]

    systems = available_systems(rows)
    if target_system not in systems:
        raise ValueError(f"{target_system} not found in {progress_path}")
    if baseline_systems:
        systems = [system for system in baseline_systems if system in systems]
    else:
        systems = [system for system in systems if system != target_system]

    predictions = {
        system: [row[system] for row in rows]
        for system in systems + [target_system]
    }

    records = []
    score_cache = {}
    for metric in metrics:
        metric_key = metric.lower()
        for system in systems + [target_system]:
            cache_key = (metric_key, system)
            if cache_key not in score_cache:
                score_cache[cache_key] = sentence_metric_scores(
                    metric_key,
                    predictions[system],
                    references,
                    sources=sources,
                    batch_size=batch_size,
                )

        target_scores = score_cache[(metric_key, target_system)]
        for baseline in systems:
            baseline_scores = score_cache[(metric_key, baseline)]
            stats = paired_bootstrap_ci(
                target_scores,
                baseline_scores,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            records.append(
                {
                    "language": lang_iso,
                    "language_name": LANG_MAP.get(lang_iso, lang_iso),
                    "metric": metric.upper(),
                    "target": target_system,
                    "baseline": baseline,
                    "target_mean": round(float(target_scores.mean()), 4),
                    "baseline_mean": round(float(baseline_scores.mean()), 4),
                    "mean_diff": round(stats["mean_diff"], 4),
                    "std_diff": round(stats["std_diff"], 4),
                    "ci_lower": round(stats["ci_lower"], 4),
                    "ci_upper": round(stats["ci_upper"], 4),
                    "p_value": round(stats["p_value"], 6),
                    "significance": significance_marker(stats["p_value"]),
                    "n": len(rows),
                    "n_bootstrap": n_bootstrap,
                }
            )
    return records


def default_progress_path(lang_iso):
    return Path(f"progress_{lang_iso}.jsonl")


def write_csv(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("No records to write.")
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(
        description="Paired bootstrap significance tests for EchoRefine outputs."
    )
    parser.add_argument("--lang", default="all", help="Language ISO code or all.")
    parser.add_argument("--progress", default=None, help="Progress JSONL for a single language.")
    parser.add_argument("--target", default="EchoRefine_Hybrid")
    parser.add_argument("--baselines", nargs="*", default=None)
    parser.add_argument("--metrics", nargs="+", default=["bleu", "chrf"])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--flores-dir", default=str(DEFAULT_LOCAL_FLORES))
    parser.add_argument(
        "--output",
        default="results/tables/significance.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    if args.lang == "all":
        lang_paths = {
            lang: default_progress_path(lang)
            for lang in LANG_MAP
            if default_progress_path(lang).exists()
        }
    else:
        progress = Path(args.progress) if args.progress else default_progress_path(args.lang)
        lang_paths = {args.lang: progress}

    all_records = []
    for lang_iso, progress_path in lang_paths.items():
        if not progress_path.exists():
            raise FileNotFoundError(progress_path)
        all_records.extend(
            analyze_language(
                lang_iso=lang_iso,
                progress_path=progress_path,
                target_system=args.target,
                baseline_systems=args.baselines,
                metrics=args.metrics,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                batch_size=args.batch_size,
                flores_dir=args.flores_dir,
            )
        )

    output_path = Path(args.output)
    write_csv(all_records, output_path)
    print(f"Wrote {len(all_records)} rows to {output_path}")


if __name__ == "__main__":
    main()
