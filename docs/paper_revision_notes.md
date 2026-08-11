# EchoRefine Paper Revision Notes

These notes map the reviewer comments to manuscript edits and repository
artifacts. The paper source was not present in this repository, so the text below
is intended to be copied into the manuscript source wherever it lives.

## Core Claim Revision

Replace "strictly dominates" and similar universal claims with:

> EchoRefine demonstrates selective improvements across languages and metrics.
> The hybrid gatekeeper improves stability by reverting unsafe refinements to the
> mBART anchor, but this conservatism can limit gains when the anchor itself is
> poor.

## Revised Abstract

We propose EchoRefine, a hybrid machine translation framework that grounds LLM
refinement in NMT anchors. EchoRefine combines mBART draft generation,
back-translation-informed LLM editing, MBR candidate selection, and
reference-free quality-estimation gating. Across eight languages on FLORES-200,
EchoRefine shows selective gains in semantic quality while exposing important
trade-offs: the gatekeeper stabilizes outputs in several settings, but can also
preserve weak anchors. We add bootstrap significance testing, component
ablations, stronger multilingual baselines, synthetic-data diagnostics, and
qualitative error analysis to identify when each component helps or hurts.

## Introduction Edits

- Present EchoRefine as a practical triadic grounding architecture rather than a
  universally dominant system.
- State that the framework is language- and metric-dependent.
- Define the "Anchor Bias Trap" as cases where conservative gating preserves a
  low-quality NMT anchor despite a potentially better refinement.
- Define the "Translationese Trap" as cases where an LLM edit becomes fluent but
  less faithful to the source/reference semantics.

## Synthetic Dataset Construction

Add the following details to Section 4.1:

1. Source-target pairs are sampled from OPUS-100 for eight English-to-X language
   directions, capped at 1,500 retained examples per language.
2. mBART-50 generates the draft translation for each English source.
3. The draft is back-translated into English with the same mBART-50 model.
4. Retained pairs must have a non-empty draft and target, draft != target,
   sentence BLEU(draft, target) < 95, and character similarity(draft, target) <
   0.98.
5. The generator records source length, draft-target sentence BLEU, and
   draft-target character similarity for diagnostics.
6. `data/validate_synthetic.py` reports edit-distance distributions,
   back-translation similarity, language counts, duplicate rates, heuristic error
   types, and optional FLORES source overlap.

## Results and Significance

Add a significance table generated with:

```bash
python analysis/significance.py --lang all --metrics bleu chrf
python analysis/significance.py --lang all --metrics comet --output results/tables/significance_comet.csv
```

Report paired bootstrap confidence intervals using N=1,000 resamples and mark
p-values as `*` for p < 0.05, `**` for p < 0.01, and `***` for p < 0.001.

## Component Ablations

Use `configs/ablations.yaml` and the new evaluator flags to report:

- mBART anchor only.
- Raw LLM refinement (`--gate-mode always_llm`).
- Full hybrid QE gate (`--gate-mode qe`).
- Greedy, beam search, and MBR decoding.
- MBR candidate counts k = 3, 5, 10, 20.
- QE margins 0.0, 0.1, 0.2, 0.3, 0.5, interpreted as the minimum CometKiwi
  advantage needed before accepting the LLM candidate.

The gatekeeper isolation table can be generated with:

```bash
python analysis/aggregate_results.py
```

## Stronger Baselines

Add at least one stronger multilingual baseline:

```bash
python run_strong_baselines.py --baseline nllb-200 --lang all
python run_strong_baselines.py --baseline aya-23 --lang all --batch-size 2
```

Recommended manuscript wording:

> We include NLLB-200 as a strong multilingual MT baseline and Aya-23 as an
> instruction-tuned multilingual LLM baseline. These comparisons test whether
> EchoRefine improves beyond contemporary multilingual systems rather than only
> weak zero-shot or undertrained direct fine-tuning baselines.

## Qualitative Error Analysis

Generate examples with:

```bash
python examples/generate_paper_examples.py
```

Add examples for:

- Accepted refinement: QE accepts an MBR-selected LLM edit.
- Over-correction prevention: QE rejects a fluent but less faithful raw edit.
- Anchor-bias candidate: QE preserves the anchor even when sentence-level chrF
  suggests the raw edit may be better.

## Equation and Model-Specification Edits

For Section 5.4, define every symbol in the QE/selection equation:

- `lambda`: interpolation weight between semantic consensus and QE score.
- `beta`: temperature or sharpness parameter used in sparsemax/soft selection.
- `sparsemax`: the sparse probability mapping from Martins and Astudillo (2016);
  cite the method directly if used.
- `q(x, y)`: CometKiwi reference-free quality score for source `x` and
  translation `y`.

If sparsemax is not implemented in the released code, remove it from the main
equation and describe the implemented hard decision rule:

```text
choose raw refinement if q(source, raw) - q(source, anchor) > margin;
otherwise choose the anchor.
```

Specify the exact Llama checkpoint as `meta-llama/Llama-3.3-70B-Instruct` and
include the model-card access date used for the experiment.

## Condensed Section 9

Replace the longer "Laws of Neural Refinement" discussion with three empirical
findings:

1. Selectivity is essential for quality: refinement should be gated rather than
   applied universally.
2. NMT anchors provide a semantic floor, but a weak floor can constrain
   improvement.
3. Component synergies are language-dependent, so decoding, candidate count, and
   QE thresholds should be tuned per setting.

## Reviewer Checklist

- [x] Repository warns against strict-dominance claims.
- [x] Significance testing script added.
- [x] Ablation config and evaluator flags added.
- [x] Raw-vs-Hybrid gatekeeper isolation supported.
- [x] Synthetic-data diagnostics added.
- [x] NLLB-200 and Aya-23 baseline wrappers added.
- [x] Qualitative example extraction added.
- [x] Missing-experiment driver added for decoding, threshold, component, and
  baseline runs.
- [x] Missing-experiment aggregation added for paper-ready ablation tables.
- [ ] Run new ablations and baseline experiments on cluster.
- [ ] Insert generated tables/figures into manuscript.
- [ ] Add permanent archive links for released data, model adapters, and scripts.
