import argparse
import json
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LANGUAGES = ["npi", "ben", "sin", "mya", "kor", "tam", "hin", "fra"]
QE_MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

EXACT_EQUIVALENCE_CLASSES = [
    ["abl_decoding_greedy", "abl_component_bt_no_mbr"],
    ["abl_decoding_mbr5", "abl_component_mbr_no_qe", "abl_threshold_0p0_always_llm"],
]

THRESHOLD_TARGETS = [
    ("abl_threshold_0p1", 0.1),
    ("abl_threshold_0p2", 0.2),
    ("abl_threshold_0p3", 0.3),
    ("abl_component_full_qe", 0.2),
]


def result_path(root, lang, run_name):
    return root / f"results_{lang}_{run_name}.json"


def progress_path(root, lang, run_name):
    return root / f"progress_{lang}_{run_name}.jsonl"


def chart_path(root, lang, run_name):
    return root / f"chart_{lang}_{run_name}.png"


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_if_missing(src, dst):
    if dst.exists() or not src.exists():
        return False
    shutil.copy2(src, dst)
    return True


def jsonl_len(path):
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def copy_longest_progress(root, lang, run_names):
    progress_counts = [
        (jsonl_len(progress_path(root, lang, run_name)), run_name)
        for run_name in run_names
    ]
    best_count, source = max(progress_counts, default=(0, None))
    if not source or best_count == 0:
        return []

    copied = []
    src = progress_path(root, lang, source)
    for target in run_names:
        dst = progress_path(root, lang, target)
        if jsonl_len(dst) >= best_count:
            continue
        shutil.copy2(src, dst)
        copied.append(f"{lang}:{source}->{target} ({best_count} rows)")
    return copied


def materialize_exact_aliases(root, languages):
    copied = []
    for lang in languages:
        for run_names in EXACT_EQUIVALENCE_CLASSES:
            source = next(
                (run for run in run_names if result_path(root, lang, run).exists()),
                None,
            )
            copied.extend(copy_longest_progress(root, lang, run_names))

            if source is not None:
                for target in run_names:
                    if target == source:
                        continue
                    wrote_result = copy_if_missing(
                        result_path(root, lang, source),
                        result_path(root, lang, target),
                    )
                    wrote_progress = copy_if_missing(
                        progress_path(root, lang, source),
                        progress_path(root, lang, target),
                    )
                    copy_if_missing(
                        chart_path(root, lang, source),
                        chart_path(root, lang, target),
                    )
                    if wrote_result or wrote_progress:
                        copied.append(f"{lang}:{source}->{target}")
    return copied


def find_mbr5_rows(root, lang):
    for run_name in EXACT_EQUIVALENCE_CLASSES[1]:
        if result_path(root, lang, run_name).exists():
            rows = read_jsonl(progress_path(root, lang, run_name))
            if rows:
                return run_name, rows
    return None, []


def load_qe_model():
    from comet import download_model, load_from_checkpoint

    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
    qe_path = download_model(QE_MODEL_NAME)
    return load_from_checkpoint(qe_path).to("cuda")


def ensure_qe_scores(rows, qe_model, batch_size):
    if all(row.get("QE_Delta") is not None for row in rows):
        return rows

    inputs = []
    for row in rows:
        inputs.append({"src": row["Source"], "mt": row["mBART"]})
        inputs.append({"src": row["Source"], "mt": row["EchoRefine_Raw"]})

    scores = qe_model.predict(
        inputs,
        batch_size=batch_size,
        gpus=1,
        progress_bar=False,
    ).scores

    scored_rows = []
    for i, row in enumerate(rows):
        updated = dict(row)
        mbart_qe = float(scores[2 * i])
        llm_qe = float(scores[2 * i + 1])
        updated["mBART_QE"] = mbart_qe
        updated["EchoRefine_Raw_QE"] = llm_qe
        updated["QE_Delta"] = llm_qe - mbart_qe
        scored_rows.append(updated)
    return scored_rows


def materialize_threshold_rows(rows, run_name, margin):
    materialized = []
    for row in rows:
        updated = dict(row)
        use_llm = updated["QE_Delta"] > margin
        updated["EchoRefine_Hybrid"] = updated["EchoRefine_Raw"] if use_llm else updated["mBART"]
        updated["Winner"] = "LLM" if use_llm else "mBART"
        updated["Gate_Mode"] = "qe"
        updated["QE_Margin"] = margin
        materialized.append(updated)
    return materialized


def materialize_anchor_rows(rows):
    materialized = []
    for row in rows:
        updated = dict(row)
        updated["EchoRefine_Hybrid"] = updated["mBART"]
        updated["Winner"] = "mBART"
        updated["Gate_Mode"] = "always_anchor"
        updated["QE_Margin"] = 0.5
        materialized.append(updated)
    return materialized


def finalize_materialized(root, lang, run_name, rows):
    from evaluate_multilingual_resumable import finalize_results

    write_jsonl(progress_path(root, lang, run_name), rows)
    srcs = [row["Source"] for row in rows]
    refs = [row["Reference"] for row in rows]
    finalize_results(rows, srcs, refs, lang, output_stem=f"{lang}_{run_name}")


def materialize_threshold_aliases(root, languages, qe_batch_size):
    written = []
    qe_model = None
    for lang in languages:
        source_name, source_rows = find_mbr5_rows(root, lang)
        if not source_rows:
            continue

        anchor_target = "abl_threshold_0p5_always_anchor"
        if not result_path(root, lang, anchor_target).exists():
            rows = materialize_anchor_rows(source_rows)
            finalize_materialized(root, lang, anchor_target, rows)
            written.append(f"{lang}:{source_name}->{anchor_target}")

        needs_qe = [
            (run_name, margin)
            for run_name, margin in THRESHOLD_TARGETS
            if not result_path(root, lang, run_name).exists()
        ]
        if not needs_qe:
            continue

        if qe_model is None:
            qe_model = load_qe_model()
        scored_rows = ensure_qe_scores(source_rows, qe_model, qe_batch_size)
        for run_name, margin in needs_qe:
            rows = materialize_threshold_rows(scored_rows, run_name, margin)
            finalize_materialized(root, lang, run_name, rows)
            written.append(f"{lang}:{source_name}->{run_name}")
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Materialize reviewer ablation aliases from reusable EchoRefine outputs."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    parser.add_argument("--qe-batch-size", type=int, default=32)
    parser.add_argument("--exact-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    os.chdir(root)
    root = Path(".")
    exact = materialize_exact_aliases(root, args.languages)
    threshold = []
    if not args.exact_only:
        threshold = materialize_threshold_aliases(root, args.languages, args.qe_batch_size)

    print(f"Materialized exact aliases: {len(exact)}")
    for item in exact:
        print(f"- {item}")
    print(f"Materialized threshold aliases: {len(threshold)}")
    for item in threshold:
        print(f"- {item}")


if __name__ == "__main__":
    main()
