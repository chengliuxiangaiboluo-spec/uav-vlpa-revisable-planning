"""Exp3: fixed, auditable input corruptions evaluated across five trained seeds."""

from __future__ import annotations

import argparse
import importlib.util
import random
import time
from pathlib import Path

from experiment_models import build_evaluation_planner
from submission_common import (
    DEFAULT_SEEDS, PROJECT_ROOT, aggregate_seed_means, evaluate, paired_seed_test, read_samples,
    run_metadata, write_json, write_rows_csv,
)

# Reuse only concrete input corruptions; all scoring remains in the shared evaluator.
_ROOT_EXP3_SPEC = importlib.util.spec_from_file_location(
    "root_exp3_corruptions", PROJECT_ROOT / "run_exp3_robustness.py"
)
if _ROOT_EXP3_SPEC is None or _ROOT_EXP3_SPEC.loader is None:
    raise RuntimeError("Cannot load the root robustness corruption utilities")
_ROOT_EXP3 = importlib.util.module_from_spec(_ROOT_EXP3_SPEC)
_ROOT_EXP3_SPEC.loader.exec_module(_ROOT_EXP3)
apply_annotation_offset = _ROOT_EXP3.apply_annotation_offset
apply_gesture_jitter = _ROOT_EXP3.apply_gesture_jitter
apply_text_perturbation = _ROOT_EXP3.apply_text_perturbation
apply_voice_noise = _ROOT_EXP3.apply_voice_noise


def perturbations(artifact_dir: Path):
    def identity(sample, _rng):
        return sample
    return {
        "clean": identity,
        "voice_snr_10db": lambda s, rng: apply_voice_noise(s, 10, rng, str(artifact_dir), "voice_snr10"),
        "gesture_jitter_6px": lambda s, rng: apply_gesture_jitter(s, 6, rng, str(artifact_dir), "gesture_jitter6"),
        "annotation_offset_10px": lambda s, rng: apply_annotation_offset(s, 10, rng, str(artifact_dir), "annotation_offset10"),
        "text_ambiguity": lambda s, rng: apply_text_perturbation(s, "ambiguous", rng),
    }


def select_smoke_samples(samples, limit: int):
    """Return a small deterministic subset that exercises every file-backed modality."""
    required_fields = ("audio_path", "gesture_image_path", "annotation_image_path")
    selected, selected_ids = [], set()
    for field in required_fields:
        candidate = next((sample for sample in samples
                          if getattr(sample, field, None)
                          and Path(getattr(sample, field)).is_file()), None)
        if candidate is None:
            raise RuntimeError(f"Smoke test cannot exercise Exp3: no readable {field} in test split")
        if candidate.scenario_id not in selected_ids:
            selected.append(candidate)
            selected_ids.add(candidate.scenario_id)
    for sample in samples:
        if len(selected) >= limit:
            break
        if sample.scenario_id not in selected_ids:
            selected.append(sample)
            selected_ids.add(sample.scenario_id)
    return selected[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3: five-seed robustness protocol")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--conditions", default="clean,voice_snr_10db,gesture_jitter_6px,annotation_offset_10px,text_ambiguity")
    parser.add_argument("--output-dir", help="Worker-specific output directory; must not already exist")
    parser.add_argument("--max-samples", type=int,
                        help="Use only the first N test samples. Intended solely for a smoke test.")
    parser.add_argument("--smoke", action="store_true",
                        help="Permit a one-seed, small-sample smoke test; never use its output in the paper.")
    parser.add_argument("--min-changed-samples", type=int, default=30,
                        help="Minimum genuinely altered inputs required for a formal corrupted condition.")
    args = parser.parse_args()
    seeds = tuple(int(part) for part in args.seeds.split(","))
    if not args.smoke and (len(seeds) != 5 or len(set(seeds)) != 5):
        raise ValueError("Exactly five distinct seeds are required")
    if args.smoke and (len(seeds) != 1 or args.max_samples is None or args.max_samples < 2):
        raise ValueError("A smoke test requires exactly one seed and --max-samples >= 2")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.min_changed_samples < 1:
        raise ValueError("--min-changed-samples must be positive")
    test = read_samples(args.test_file)
    if args.smoke:
        test = select_smoke_samples(test, args.max_samples)
    elif args.max_samples is not None:
        test = test[:args.max_samples]
    condition_names = {"clean", "voice_snr_10db", "gesture_jitter_6px", "annotation_offset_10px", "text_ambiguity"}
    selected = tuple(part.strip() for part in args.conditions.split(",") if part.strip())
    if not selected or any(name not in condition_names for name in selected):
        raise ValueError(f"conditions must be selected from {','.join(sorted(condition_names))}")
    output = Path(args.output_dir).resolve() if args.output_dir else Path(args.run_root).resolve() / "results" / "exp3_robustness"
    artifacts = output / "corrupted_inputs"
    output.mkdir(parents=True, exist_ok=False)
    artifacts.mkdir()
    started_at = time.monotonic()
    print(f"[Exp3] start | test_samples={len(test)} | conditions={selected} | seeds={seeds}", flush=True)

    all_rows, per_condition, perturbation_audit = [], {}, {}
    paired_clean_rows, paired_clean_summary = {}, {}
    for condition, transform in perturbations(artifacts).items():
        if condition not in selected:
            continue
        # The same deterministic corruptions are used for every trained seed.
        rng = random.Random(f"submission-2026-09-10:{condition}")
        corrupted = [transform(sample, rng) for sample in test]
        def was_changed(source, altered):
            return source is not altered and any(
                getattr(source, field, None) != getattr(altered, field, None)
                for field in ("audio_path", "gesture_image_path", "annotation_image_path", "text_instruction")
            )
        changed_mask = [was_changed(source, altered) for source, altered in zip(test, corrupted)]
        changed = sum(changed_mask)
        if condition != "clean" and changed == 0:
            raise RuntimeError(f"Corruption {condition} changed zero inputs; refusing a no-op robustness claim")
        if condition != "clean" and not args.smoke and changed < args.min_changed_samples:
            raise RuntimeError(
                f"Corruption {condition} changed only {changed} inputs; need at least "
                f"{args.min_changed_samples} for a reportable robustness condition"
            )
        perturbation_audit[condition] = {"n_inputs": len(test), "n_changed": changed}
        print(f"[Exp3] corruption prepared | condition={condition} | changed={changed}/{len(test)}", flush=True)
        per_condition[condition] = {}
        paired_clean_rows[condition] = {}
        # A corruption must be compared with clean execution on exactly the
        # same eligible scenarios.  Otherwise scenes without the modality
        # dilute the effect and can create an invalid apparent improvement.
        source_for_evaluation = test if condition == "clean" else [
            source for source, changed_here in zip(test, changed_mask) if changed_here
        ]
        corrupted_for_evaluation = corrupted if condition == "clean" else [
            altered for altered, changed_here in zip(corrupted, changed_mask) if changed_here
        ]
        for seed in seeds:
            print(f"[Exp3] evaluation started | condition={condition} | seed={seed}", flush=True)
            planner, _ = build_evaluation_planner("ours", args.run_root, seed)
            if condition != "clean":
                clean_rows = evaluate(planner, source_for_evaluation, planner.benchmark_dir)
                paired_clean_rows[condition][seed] = clean_rows
            rows = evaluate(planner, corrupted_for_evaluation, planner.benchmark_dir)
            for row in rows:
                row.update({"condition": condition, "training_seed": seed})
            per_condition[condition][seed] = rows
            all_rows.extend(rows)
            print(f"[Exp3] evaluation complete | condition={condition} | seed={seed} | rows={len(rows)}", flush=True)

    summary = {name: aggregate_seed_means(rows) for name, rows in per_condition.items()}
    comparisons = {name: paired_seed_test(paired_clean_rows[name], rows)
                   for name, rows in per_condition.items() if name != "clean"}
    paired_clean_summary = {
        name: aggregate_seed_means(rows)
        for name, rows in paired_clean_rows.items() if name != "clean"
    }
    write_rows_csv(output / "per_scenario_results.csv", all_rows)
    write_json(output / "summary.json", run_metadata({
        "experiment": "Exp3 fixed input-corruption robustness",
        "test_file": str(Path(args.test_file).resolve()),
        "seeds": list(seeds),
        "smoke_test": args.smoke,
        "max_samples": args.max_samples,
        "min_changed_samples": args.min_changed_samples,
        "worker_conditions": list(selected),
        "perturbation_audit": perturbation_audit,
        "summary": summary,
        "paired_clean_reference_summary": paired_clean_summary,
        "paired_tests_vs_clean": comparisons,
        "claim_limit": "robustness to the listed synthetic input corruptions on their modality-available paired subsets only; not real-flight robustness",
    }))
    print(f"[Exp3] complete | elapsed={(time.monotonic() - started_at) / 60.0:.1f} min | output={output}", flush=True)


if __name__ == "__main__":
    main()
