import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


LANGUAGES = ["npi", "ben", "sin", "mya", "kor", "tam", "hin", "fra"]
EVALUATOR = "evaluate_multilingual_resumable.py"


@dataclass(frozen=True)
class Experiment:
    group: str
    name: str
    description: str
    args: tuple[str, ...]
    requires_lang: bool = True
    result_pattern: str | None = None

    def result_path(self, lang=None):
        if not self.result_pattern:
            return None
        if self.requires_lang:
            return Path(self.result_pattern.format(lang=lang, name=self.name))
        return Path(self.result_pattern.format(name=self.name))


def evaluator_experiment(group, name, description, *args):
    return Experiment(
        group=group,
        name=name,
        description=description,
        args=("--run-name", name, *args),
        result_pattern="results_{lang}_{name}.json",
    )


DECODING_EXPERIMENTS = [
    evaluator_experiment(
        "decoding",
        "abl_decoding_greedy",
        "Greedy refinement without MBR or QE.",
        "--decoding-method", "greedy",
        "--num-candidates", "1",
        "--gate-mode", "always_llm",
    ),
    evaluator_experiment(
        "decoding",
        "abl_decoding_beam4",
        "Beam-search refinement with 4 beams, no QE.",
        "--decoding-method", "beam_search_4",
        "--num-candidates", "4",
        "--gate-mode", "always_llm",
    ),
    evaluator_experiment(
        "decoding",
        "abl_decoding_beam8",
        "Beam-search refinement with 8 beams, no QE.",
        "--decoding-method", "beam_search_8",
        "--num-candidates", "8",
        "--gate-mode", "always_llm",
    ),
    evaluator_experiment(
        "decoding",
        "abl_decoding_mbr3",
        "MBR refinement with 3 sampled candidates, no QE.",
        "--decoding-method", "mbr",
        "--num-candidates", "3",
        "--gate-mode", "always_llm",
    ),
    evaluator_experiment(
        "decoding",
        "abl_decoding_mbr5",
        "MBR refinement with 5 sampled candidates, no QE.",
        "--decoding-method", "mbr",
        "--num-candidates", "5",
        "--gate-mode", "always_llm",
    ),
    evaluator_experiment(
        "decoding",
        "abl_decoding_mbr10",
        "MBR refinement with 10 sampled candidates, no QE.",
        "--decoding-method", "mbr",
        "--num-candidates", "10",
        "--gate-mode", "always_llm",
    ),
    evaluator_experiment(
        "decoding",
        "abl_decoding_mbr20",
        "MBR refinement with 20 sampled candidates, no QE.",
        "--decoding-method", "mbr",
        "--num-candidates", "20",
        "--gate-mode", "always_llm",
    ),
]


THRESHOLD_EXPERIMENTS = [
    evaluator_experiment(
        "thresholds",
        "abl_threshold_0p0_always_llm",
        "Threshold endpoint: always accept the LLM refinement.",
        "--decoding-method", "mbr",
        "--num-candidates", "5",
        "--gate-mode", "always_llm",
        "--qe-margin", "0.0",
    ),
    evaluator_experiment(
        "thresholds",
        "abl_threshold_0p1",
        "QE margin 0.1.",
        "--decoding-method", "mbr",
        "--num-candidates", "5",
        "--gate-mode", "qe",
        "--qe-margin", "0.1",
    ),
    evaluator_experiment(
        "thresholds",
        "abl_threshold_0p2",
        "QE margin 0.2.",
        "--decoding-method", "mbr",
        "--num-candidates", "5",
        "--gate-mode", "qe",
        "--qe-margin", "0.2",
    ),
    evaluator_experiment(
        "thresholds",
        "abl_threshold_0p3",
        "QE margin 0.3.",
        "--decoding-method", "mbr",
        "--num-candidates", "5",
        "--gate-mode", "qe",
        "--qe-margin", "0.3",
    ),
    evaluator_experiment(
        "thresholds",
        "abl_threshold_0p5_always_anchor",
        "Threshold endpoint: always keep the anchor.",
        "--decoding-method", "mbr",
        "--num-candidates", "5",
        "--gate-mode", "always_anchor",
        "--qe-margin", "0.5",
    ),
]


COMPONENT_EXPERIMENTS = [
    evaluator_experiment(
        "components",
        "abl_component_no_bt",
        "LLM refinement without back-translation signal, greedy decoding, no QE.",
        "--decoding-method", "greedy",
        "--num-candidates", "1",
        "--gate-mode", "always_llm",
        "--no-backtranslation",
    ),
    evaluator_experiment(
        "components",
        "abl_component_bt_no_mbr",
        "Back-translation signal with greedy decoding, no MBR, no QE.",
        "--decoding-method", "greedy",
        "--num-candidates", "1",
        "--gate-mode", "always_llm",
    ),
    evaluator_experiment(
        "components",
        "abl_component_mbr_no_qe",
        "Back-translation plus MBR, no QE gate.",
        "--decoding-method", "mbr",
        "--num-candidates", "5",
        "--gate-mode", "always_llm",
    ),
    evaluator_experiment(
        "components",
        "abl_component_full_qe",
        "Full EchoRefine with back-translation, MBR, and QE margin 0.2.",
        "--decoding-method", "mbr",
        "--num-candidates", "5",
        "--gate-mode", "qe",
        "--qe-margin", "0.2",
    ),
]


BASELINE_EXPERIMENTS = [
    Experiment(
        group="baselines",
        name="nllb200",
        description="NLLB-200 distilled multilingual MT baseline.",
        args=(
            "run_strong_baselines.py",
            "--baseline", "nllb-200",
            "--lang", "all",
            "--progress-dir", "results/strong_baselines",
            "--output-dir", "results/strong_baselines",
            "--skip-existing",
        ),
        requires_lang=False,
        result_pattern=None,
    ),
    Experiment(
        group="baselines",
        name="aya23",
        description="Aya-23-35B prompted multilingual baseline.",
        args=(
            "run_strong_baselines.py",
            "--baseline", "aya-23",
            "--lang", "all",
            "--batch-size", "2",
            "--progress-dir", "results/strong_baselines",
            "--output-dir", "results/strong_baselines",
            "--skip-existing",
        ),
        requires_lang=False,
        result_pattern=None,
    ),
]


ALL_EXPERIMENTS = [
    *DECODING_EXPERIMENTS,
    *THRESHOLD_EXPERIMENTS,
    *COMPONENT_EXPERIMENTS,
    *BASELINE_EXPERIMENTS,
]


def selected_experiments(groups, experiment_names=None):
    requested = set(groups)
    if "all" in requested:
        experiments = ALL_EXPERIMENTS
    else:
        if "high-priority" in requested:
            requested.update({"decoding", "thresholds", "baselines"})
        experiments = [experiment for experiment in ALL_EXPERIMENTS if experiment.group in requested]

    if experiment_names:
        wanted = set(experiment_names)
        experiments = [experiment for experiment in experiments if experiment.name in wanted]
        missing = wanted - {experiment.name for experiment in experiments}
        if missing:
            raise SystemExit(f"Unknown or unselected experiments: {', '.join(sorted(missing))}")
    return experiments


def command_for_experiment(
    experiment,
    python_bin,
    lang,
    limit,
    batch_size,
    extra_args,
    include_auxiliary_generations,
):
    if experiment.group == "baselines":
        return [python_bin, *experiment.args]

    command = [
        python_bin,
        EVALUATOR,
        "--lang",
        lang,
        "--batch-size",
        str(batch_size),
        *experiment.args,
        *extra_args,
    ]
    if not include_auxiliary_generations:
        command.append("--skip-auxiliary-generations")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def iter_commands(
    experiments,
    languages,
    python_bin,
    limit,
    batch_size,
    extra_args,
    include_auxiliary_generations,
):
    for experiment in experiments:
        if experiment.requires_lang:
            for lang in languages:
                yield experiment, lang, command_for_experiment(
                    experiment,
                    python_bin,
                    lang,
                    limit,
                    batch_size,
                    extra_args,
                    include_auxiliary_generations,
                )
        else:
            yield experiment, None, command_for_experiment(
                experiment,
                python_bin,
                None,
                limit,
                batch_size,
                extra_args,
                include_auxiliary_generations,
            )


def run_command(command, dry_run):
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run reviewer-requested EchoRefine ablations and stronger baselines."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["high-priority"],
        choices=["all", "high-priority", "decoding", "thresholds", "components", "baselines"],
    )
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    parser.add_argument("--limit", type=int, default=1012)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Optional exact experiment names to run, e.g. abl_decoding_mbr5.",
    )
    parser.add_argument(
        "--include-auxiliary-generations",
        action="store_true",
        help="Also generate zero-shot and direct-adapter outputs for evaluator experiments.",
    )
    parser.add_argument(
        "--extra-evaluator-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional args appended to each evaluator command after '--'.",
    )
    args = parser.parse_args()

    experiments = selected_experiments(args.groups, args.experiments)
    if not experiments:
        raise SystemExit("No experiments selected.")

    failures = []
    for experiment, lang, command in iter_commands(
        experiments,
        args.languages,
        args.python_bin,
        args.limit,
        args.batch_size,
        args.extra_evaluator_args,
        args.include_auxiliary_generations,
    ):
        result_path = experiment.result_path(lang)
        if args.skip_existing and result_path and result_path.exists():
            print(f"Skipping {experiment.name} {lang or ''}: {result_path} exists")
            continue
        code = run_command(command, args.dry_run)
        if code != 0:
            failures.append((experiment.name, lang, code))
            print(f"FAILED {experiment.name} {lang or ''}: exit {code}", flush=True)

    if failures:
        print("Failures:")
        for name, lang, code in failures:
            print(f"- {name} {lang or ''}: exit {code}")
        raise SystemExit(1)

    print("All selected experiments completed.")


if __name__ == "__main__":
    main()
