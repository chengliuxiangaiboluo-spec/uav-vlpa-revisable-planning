# Reproducibility guide

## Scope of this public release

This repository releases the implementation, protocol runners, preflight checks, and release-integrity checker associated with the manuscript. It does not redistribute model weights, checkpoints, benchmark imagery, map tiles, audio, GPS trajectories, raw model responses, or other controlled assets. Consequently, a fresh clone can verify the released source and inspect every public execution path, but it cannot reproduce the reported numerical tables without the authorised assets listed in `ASSET_MANIFEST.template.yaml`.

The reported synthetic, OSM-grounding, and fixed-input GPS protocols have different inputs and evaluation units. They must be executed and interpreted separately. The frozen UAV-VLPA* Molmo adaptation is a deterministic single-run reference, whereas learned systems use five independently trained seeds.

## Environment and release-integrity check

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

The final command compiles the released Python sources and checks for common private-path markers and protected artefacts. It is not an end-to-end reproduction claim.

## Authorised assets required for full re-execution

Copy `ASSET_MANIFEST.template.yaml` outside the Git checkout and complete it with authorised local paths and checksums. The following asset families are required:

| Asset family | Required for | Public in this repository |
| --- | --- | --- |
| UAV-VLPA benchmark and split manifests | Strict OSM protocol construction, five-seed training, and held-out evaluation | No |
| Pretrained model cache | Offline text, audio, vision, and Molmo components | No |
| Seed checkpoints | Exact re-evaluation of a trained seed | No |
| Controlled GPS data | Fixed-input GPS descriptive check | No |

Do not use reference coordinates as planner inputs. They are retained on evaluator-side copies only for scoring.

## Protocol entry points

| Purpose | Entry point |
| --- | --- |
| Prepare strict OSM V2 protocol | `submission_experiments_20260910/prepare_v2_osm_protocol.py` |
| Train a seed | `submission_experiments_20260910/train_submission_seed.py` |
| Run V2 grounding and replanning | `submission_experiments_20260910/run_v2_grounding_and_replan.py` |
| Run frozen UAV-VLPA* reference | `submission_experiments_20260910/run_uav_vlpa_molmo_v2.py` |
| Run Exp1 local fusion controls | `submission_experiments_20260910/run_exp1_fair_baselines.py` |
| Run Exp5 ablation | `submission_experiments_20260910/run_exp5_architecture_ablation.py` |
| Run availability-control analysis | `submission_experiments_20260910/run_beta_coupling_control.py` |
| Validate and merge generated run outputs | `submission_experiments_20260910/validate_submission_run.py` and the corresponding `merge_*.py` script |

Each runner writes protocol-specific outputs. Do not pool values across protocols, and preserve the generated per-scenario rows, split records, training metadata, and checkpoint-selection metadata for any new run.

## Reporting boundary

The optional LoRA prototype is isolated in `experimental_not_used_in_paper/` and is not part of any result reported in the manuscript. The released scripts do not call a hosted LLM during paper-consistent evaluation. The paper-consistent grounding path uses frozen 4-bit Molmo-7B-O inference offline.

For the supplementary evidence archive, split checksums, prompt and decoding configuration, parser audits, planned-contrast statistics, and the restricted-data inclusion/exclusion manifest are documented separately from this public code release. Report the commit SHA and the relevant protocol name when requesting help or reporting a reproduction issue.
