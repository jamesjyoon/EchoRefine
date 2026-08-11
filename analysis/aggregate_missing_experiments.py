import argparse
import json
from pathlib import Path

import pandas as pd


LANGUAGES = ["npi", "ben", "sin", "mya", "kor", "tam", "hin", "fra"]

DECODING_CONFIGS = [
    ("Greedy", "abl_decoding_greedy"),
    ("Beam Search (k=4)", "abl_decoding_beam4"),
    ("Beam Search (k=8)", "abl_decoding_beam8"),
    ("MBR (k=3)", "abl_decoding_mbr3"),
    ("MBR (k=5)", "abl_decoding_mbr5"),
    ("MBR (k=10)", "abl_decoding_mbr10"),
    ("MBR (k=20)", "abl_decoding_mbr20"),
]

THRESHOLD_CONFIGS = [
    ("0.0 (always LLM)", "abl_threshold_0p0_always_llm"),
    ("0.1", "abl_threshold_0p1"),
    ("0.2", "abl_threshold_0p2"),
    ("0.3", "abl_threshold_0p3"),
    ("0.5 (always anchor)", "abl_threshold_0p5_always_anchor"),
]

COMPONENT_CONFIGS = [
    ("mBART Anchor", None, "mBART"),
    ("+ LLM Refinement (no BT)", "abl_component_no_bt", "EchoRefine_Hybrid"),
    ("+ Back-translation Signal", "abl_component_bt_no_mbr", "EchoRefine_Hybrid"),
    ("+ MBR (k=5)", "abl_component_mbr_no_qe", "EchoRefine_Hybrid"),
    ("+ QE Gate (Hybrid)", None, "EchoRefine_Hybrid"),
]

BASELINE_CONFIGS = [
    ("NLLB-200 distilled", Path("results/strong_baselines/results_nllb200_{lang}.json"), "NLLB200", "600M"),
    ("Aya-23-35B", Path("results/strong_baselines/results_aya23_{lang}.json"), "Aya23", "35B"),
    ("EchoRefine Hybrid", Path("results_{lang}.json"), "EchoRefine_Hybrid", "70B"),
]


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
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


def result_path(root, lang, run_name):
    if run_name is None:
        return root / f"results_{lang}.json"
    return root / f"results_{lang}_{run_name}.json"


def progress_path(root, lang, run_name):
    if run_name is None:
        return root / f"progress_{lang}.jsonl"
    return root / f"progress_{lang}_{run_name}.jsonl"


def collect_system_records(root, configs, config_label, output_system="EchoRefine_Hybrid"):
    records = []
    missing = []
    for label, run_name in configs:
        for lang in LANGUAGES:
            path = result_path(root, lang, run_name)
            if not path.exists():
                missing.append(str(path))
                continue
            metrics = read_json(path)
            system = output_system
            if system not in metrics:
                missing.append(f"{path}:{system}")
                continue
            record = {
                config_label: label,
                "language": lang,
                **metrics[system],
            }
            rows = read_jsonl(progress_path(root, lang, run_name))
            if rows:
                llm_wins = sum(row.get("Winner") == "LLM" for row in rows)
                record["n"] = len(rows)
                record["llm_wins"] = llm_wins
                record["anchor_wins"] = sum(row.get("Winner") == "mBART" for row in rows)
                record["llm_win_rate"] = round(llm_wins / len(rows), 4)
            records.append(record)
    return pd.DataFrame(records), missing


def collect_component_records(root):
    records = []
    missing = []
    for label, run_name, system in COMPONENT_CONFIGS:
        for lang in LANGUAGES:
            path = result_path(root, lang, run_name)
            if not path.exists():
                missing.append(str(path))
                continue
            metrics = read_json(path)
            if system not in metrics:
                missing.append(f"{path}:{system}")
                continue
            records.append(
                {
                    "component_configuration": label,
                    "language": lang,
                    **metrics[system],
                }
            )
    return pd.DataFrame(records), missing


def collect_strong_baseline_records(root):
    records = []
    missing = []
    for label, pattern, system, model_size in BASELINE_CONFIGS:
        for lang in LANGUAGES:
            path = root / str(pattern).format(lang=lang)
            if not path.exists():
                missing.append(str(path))
                continue
            metrics = read_json(path)
            if system not in metrics:
                missing.append(f"{path}:{system}")
                continue
            records.append(
                {
                    "system": label,
                    "language": lang,
                    "model_size": model_size,
                    **metrics[system],
                }
            )
    return pd.DataFrame(records), missing


def macro_average(df, label_col):
    if df.empty:
        return df
    metric_cols = [col for col in ["BLEU", "chrF", "COMET", "llm_win_rate"] if col in df.columns]
    return df.groupby(label_col, as_index=False)[metric_cols].mean().round(4)


def write_table(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Aggregate reviewer-requested ablation outputs.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_records = {}

    decoding_df, missing_records["decoding"] = collect_system_records(
        root,
        DECODING_CONFIGS,
        "decoding_method",
    )
    if not decoding_df.empty:
        write_table(decoding_df, output_dir / "decoding_ablation_by_language.csv")
        write_table(macro_average(decoding_df, "decoding_method"), output_dir / "decoding_ablation_macro.csv")

    threshold_df, missing_records["thresholds"] = collect_system_records(
        root,
        THRESHOLD_CONFIGS,
        "threshold",
    )
    if not threshold_df.empty:
        write_table(threshold_df, output_dir / "threshold_sensitivity_by_language.csv")
        write_table(macro_average(threshold_df, "threshold"), output_dir / "threshold_sensitivity_macro.csv")

    component_df, missing_records["components"] = collect_component_records(root)
    if not component_df.empty:
        write_table(component_df, output_dir / "component_ablation_by_language.csv")
        write_table(
            macro_average(component_df, "component_configuration"),
            output_dir / "component_ablation_macro.csv",
        )

    baseline_df, missing_records["baselines"] = collect_strong_baseline_records(root)
    if not baseline_df.empty:
        write_table(baseline_df, output_dir / "strong_baselines_by_language.csv")
        baseline_macro = macro_average(baseline_df, "system")
        sizes = baseline_df[["system", "model_size"]].drop_duplicates()
        baseline_macro = baseline_macro.merge(sizes, on="system", how="left")
        write_table(baseline_macro, output_dir / "strong_baselines_macro.csv")

    missing_path = output_dir / "missing_experiment_outputs.json"
    missing_path.write_text(json.dumps(missing_records, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote ablation tables to {output_dir}")
    print(f"Wrote missing-output manifest to {missing_path}")


if __name__ == "__main__":
    main()
