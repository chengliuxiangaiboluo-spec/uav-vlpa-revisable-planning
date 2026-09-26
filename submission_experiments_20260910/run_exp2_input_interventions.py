"""Exp2: controlled test-time modality interventions across five trained seeds.

This is intentionally named an input-intervention study, not a causal
architecture-ablation study.  Architecture causal claims require separately
trained, configuration-specific models and are not fabricated here.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from submission_common import (
    DEFAULT_SEEDS, InputInterventionPlanner, aggregate_seed_means, evaluate,
    paired_seed_test, read_samples, run_metadata, write_json, write_rows_csv,
)
from data.scenario_schema import ModalityType
from experiment_models import build_evaluation_planner

CONDITIONS = {
    "full_available": {ModalityType.TEXT, ModalityType.VOICE, ModalityType.GESTURE, ModalityType.ANNOTATION},
    "text_only": {ModalityType.TEXT},
    "minus_voice": {ModalityType.TEXT, ModalityType.GESTURE, ModalityType.ANNOTATION},
    "minus_gesture": {ModalityType.TEXT, ModalityType.VOICE, ModalityType.ANNOTATION},
    "minus_annotation": {ModalityType.TEXT, ModalityType.VOICE, ModalityType.GESTURE},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2: strict modality input interventions")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--output-dir", help="Worker-specific output directory; must not already exist")
    parser.add_argument("--max-samples", type=int, help="Use only for a preflight smoke test")
    parser.add_argument("--smoke", action="store_true", help="Permit a one-seed, few-sample preflight only")
    parser.add_argument("--require-task-trace", action="store_true", help="Fail if IA cannot see the planner task trace")
    parser.add_argument("--require-modality", choices=("voice", "gesture", "annotation"),
                        help="Smoke-only: retain samples with a usable required modality input")
    args = parser.parse_args()
    seeds = tuple(int(part) for part in args.seeds.split(","))
    if (not args.smoke and (len(seeds) != 5 or len(set(seeds)) != 5)) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Exactly five distinct seeds are required")
    test = read_samples(args.test_file)
    if args.require_modality:
        if not args.smoke:
            raise ValueError("--require-modality is permitted only for a --smoke run")
        modality = ModalityType(args.require_modality)
        field = {"voice": "audio_path", "gesture": "gesture_image_path", "annotation": "annotation_image_path"}[args.require_modality]
        test = [sample for sample in test if modality in sample.modalities and getattr(sample, field, None)]
        if not test:
            raise RuntimeError(f"Smoke preflight found no usable {args.require_modality} samples")
    if args.max_samples is not None:
        if not args.smoke or args.max_samples < 1:
            raise ValueError("--max-samples is permitted only for a positive --smoke run")
        test = test[:args.max_samples]
    selected = tuple(part.strip() for part in args.conditions.split(",") if part.strip())
    if not selected or any(name not in CONDITIONS for name in selected):
        raise ValueError(f"conditions must be selected from {','.join(CONDITIONS)}")
    output = Path(args.output_dir).resolve() if args.output_dir else Path(args.run_root).resolve() / "results" / "exp2_input_interventions"
    output.mkdir(parents=True, exist_ok=False)
    started_at = time.monotonic()
    print(f"[Exp2] start | test_samples={len(test)} | conditions={selected} | seeds={seeds}", flush=True)

    per_condition, all_rows, checkpoint_audit = {}, [], {}
    for condition in selected:
        modalities = CONDITIONS[condition]
        per_condition[condition] = {}
        for seed in seeds:
            print(f"[Exp2] evaluation started | condition={condition} | seed={seed}", flush=True)
            planner, checkpoints = build_evaluation_planner("ours", args.run_root, seed)
            checkpoint_audit[str(seed)] = checkpoints
            intervention = InputInterventionPlanner(planner, modalities)
            rows = evaluate(intervention, test, planner.benchmark_dir)
            if args.require_task_trace and not intervention.last_tasks:
                raise RuntimeError(
                    "Instruction-accuracy preflight failed: wrapper received no task trace from the planner"
                )
            if args.require_task_trace and any("instruction_accuracy" not in row for row in rows):
                raise RuntimeError("Instruction-accuracy preflight failed: metric missing from result rows")
            if args.require_modality and condition.startswith("minus_"):
                removed = condition[len("minus_"):]
                if removed == args.require_modality and args.require_modality in intervention.last_delivered_modalities:
                    raise RuntimeError(f"Intervention preflight failed: {args.require_modality} was still delivered")
            for row in rows:
                row.update({"condition": condition, "training_seed": seed})
            per_condition[condition][seed] = rows
            all_rows.extend(rows)
            print(f"[Exp2] evaluation complete | condition={condition} | seed={seed} | rows={len(rows)}", flush=True)

    summary = {name: aggregate_seed_means(rows) for name, rows in per_condition.items()}
    tests = ({name: paired_seed_test(per_condition["full_available"], rows)
              for name, rows in per_condition.items() if name != "full_available"}
             if "full_available" in per_condition else {})
    write_rows_csv(output / "per_scenario_results.csv", all_rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "Exp2 controlled modality input interventions",
        "test_file": str(Path(args.test_file).resolve()),
        "seeds": list(seeds),
        "worker_conditions": list(selected),
        "conditions": {key: [m.value for m in value] for key, value in CONDITIONS.items()},
        "claim_allowed": "test-time dependence on available input modalities under a fixed trained full model",
        "claim_forbidden": "causal contribution of an architecture component or independently trained modality-specific model",
        "checkpoint_audit": checkpoint_audit,
        "smoke_test": bool(args.smoke),
        "required_smoke_modality": args.require_modality,
        "summary": summary,
        "paired_tests_vs_full_available": tests,
    }))
    print(f"[Exp2] complete | elapsed={(time.monotonic() - started_at) / 60.0:.1f} min | output={output}", flush=True)


if __name__ == "__main__":
    main()
