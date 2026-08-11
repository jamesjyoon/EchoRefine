import argparse
import json
from pathlib import Path

import pandas as pd


LANGS = ["npi", "ben", "sin", "mya", "kor", "tam", "hin", "fra"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
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


def aggregate_metrics(results_dir):
    records = []
    for lang in LANGS:
        path = results_dir / f"results_{lang}.json"
        if not path.exists():
            continue
        metrics = load_json(path)
        for system, values in metrics.items():
            record = {"language": lang, "system": system}
            record.update(values)
            records.append(record)
    return pd.DataFrame(records)


def gatekeeper_isolation(results_dir):
    records = []
    for lang in LANGS:
        result_path = results_dir / f"results_{lang}.json"
        progress_path = results_dir / f"progress_{lang}.jsonl"
        if not result_path.exists():
            continue
        metrics = load_json(result_path)
        rows = load_jsonl(progress_path)
        winner_counts = pd.Series([row.get("Winner", "unknown") for row in rows]).value_counts().to_dict()
        raw = metrics.get("EchoRefine_Raw", {})
        hybrid = metrics.get("EchoRefine_Hybrid", {})
        anchor = metrics.get("mBART", {})
        record = {
            "language": lang,
            "n": len(rows),
            "llm_wins": int(winner_counts.get("LLM", 0)),
            "anchor_wins": int(winner_counts.get("mBART", 0)),
        }
        for metric in ["BLEU", "chrF", "COMET"]:
            record[f"raw_{metric}"] = raw.get(metric)
            record[f"hybrid_{metric}"] = hybrid.get(metric)
            record[f"anchor_{metric}"] = anchor.get(metric)
            if raw.get(metric) is not None and hybrid.get(metric) is not None:
                record[f"hybrid_minus_raw_{metric}"] = round(hybrid[metric] - raw[metric], 2)
            if anchor.get(metric) is not None and hybrid.get(metric) is not None:
                record[f"hybrid_minus_anchor_{metric}"] = round(hybrid[metric] - anchor[metric], 2)
        records.append(record)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Aggregate EchoRefine result JSON files into CSV tables.")
    parser.add_argument("--results-dir", default=".")
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = aggregate_metrics(results_dir)
    if not metrics_df.empty:
        metrics_df.to_csv(output_dir / "table2_results.csv", index=False)
        averages = metrics_df.groupby("system", as_index=False)[["BLEU", "chrF", "COMET"]].mean()
        averages.to_csv(output_dir / "table2_macro_averages.csv", index=False)

    gate_df = gatekeeper_isolation(results_dir)
    if not gate_df.empty:
        gate_df.to_csv(output_dir / "hybrid_gatekeeper_isolation.csv", index=False)

    print(f"Wrote aggregate tables to {output_dir}")


if __name__ == "__main__":
    main()
