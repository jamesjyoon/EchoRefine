# Synthetic Dataset Diagnostics

Dataset: `train_data_multilingual.csv`
Examples: 12000

## Language Counts

- Bengali: 1500
- Burmese: 1500
- French: 1500
- Hindi: 1500
- Korean: 1500
- Nepali: 1500
- Sinhala: 1500
- Tamil: 1500

## Edit Distance

{
  "mean": 0.8119,
  "std": 0.233,
  "p05": 0.3333,
  "p25": 0.6667,
  "p50": 0.9091,
  "p75": 1.0,
  "p95": 1.0
}

## Back-Translation Similarity

{
  "mean": 0.715,
  "std": 0.2499,
  "p05": 0.2222,
  "p25": 0.5556,
  "p50": 0.7586,
  "p75": 0.9333,
  "p95": 1.0
}

## Error Type Counts

- length_shift: 5367
- lexical_or_morphological_edit: 1136
- major_rewrite: 5287
- near_duplicate: 1
- source_copy: 209
