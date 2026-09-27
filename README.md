# Modality-Aware Coupling for Multimodal UAV Planning

This repository contains the public, audit-oriented implementation associated with *Modality-Aware Coupling with a Revisable Instruction-to-Execution Interface for Multimodal UAV Planning*. It implements multimodal fusion, hierarchical semantic-aware task decomposition (HSATD), availability-aware dynamic replanning (AADR), and the experiment runners used to produce the paper's protocol-specific results.

## Release boundary

This is a **code release**, not a redistribution of data, model weights, checkpoints, GPS trajectories, satellite imagery, audio, or raw model responses. Consequently, a fresh clone can verify source integrity and run the supplied preflight checks immediately, but exact numerical reproduction requires the controlled assets listed in [`reproducibility/ASSET_MANIFEST.template.yaml`](reproducibility/ASSET_MANIFEST.template.yaml).

The paper reports distinct synthetic, OSM-grounding, and fixed-input GPS protocols. Their results must not be pooled. The frozen UAV-VLPA* Molmo adaptation is evaluated as a deterministic single-run reference, whereas learned methods use five independently trained seeds.

## Paper-consistent execution path

The reported experiments use frozen 4-bit Molmo-7B-O inference for offline map grounding. They do **not** fine-tune Molmo with LoRA or call a hosted LLM during an evaluation run. The default code path therefore disables external LLM enrichment. The retained LoRA prototype is isolated in [`experimental_not_used_in_paper/`](experimental_not_used_in_paper/) and is not part of any result reported in the manuscript.

## Quick start

```bash
git clone https://github.com/chengliuxiangaiboluo-spec/uav-vlpa-revisable-planning.git
cd uav-vlpa-revisable-planning
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

The final command compiles every shipped Python module and rejects accidental host paths or protected artefacts. It is a release-integrity check; it does not claim to reproduce a trained result without the declared assets.

## Re-executing the paper protocols

1. Obtain the authorized benchmark, split manifests, model cache, and any checkpoint bundle described in the asset manifest. Do not use label coordinates as planner input.
2. Point the checkout to the local model cache:

   ```bash
   export UAV_VLPA_WEIGHTS_DIR=/absolute/path/to/Weights
   ```

3. Prepare the strict OSM protocol from an empty run directory:

   ```bash
   python submission_experiments_20260910/prepare_v2_osm_protocol.py \
     --run-root /absolute/path/to/run_root --cache-only
   ```

4. Train one independently seeded model and retain its metadata:

   ```bash
   python submission_experiments_20260910/train_submission_seed.py \
     --method ours --seed 42 \
     --train-file /absolute/path/to/run_root/splits/train.json \
     --val-file /absolute/path/to/run_root/splits/val_location.json \
     --run-root /absolute/path/to/run_root --server
   ```

5. Use the corresponding scripts in `submission_experiments_20260910/` for each reported protocol. The runners fail rather than overwrite outputs, and preserve per-scenario rows and checkpoint/protocol audit metadata.

## What is included

- `models/`, `training/`, `evaluation/`, `data/`, `utils/`, and `configs/`: implementation code only; no generated dataset files;
- `submission_experiments_20260910/`: the strict experiment runners, aggregation, and preflight code used for the paper; and
- `scripts/verify_release.py`: a portable public-release integrity check.

The default [`requirements.txt`](requirements.txt) lists the dependencies for the paper-consistent execution path. Optional packages for retained, non-paper experimental modules are listed separately in [`experimental_not_used_in_paper/requirements-experimental.txt`](experimental_not_used_in_paper/requirements-experimental.txt).

Host-specific paths have been replaced by the `UAV_VLPA_PROJECT_ROOT` and `UAV_VLPA_WEIGHTS_DIR` environment variables.

## Citation and licence

Until the associated manuscript receives its final bibliographic record, cite this repository by its stable URL and the release tag recorded for the submitted version. A `CITATION.cff` file and a reuse licence will be added only after all authors approve the legal and bibliographic metadata. Until then, this repository is provided for inspection and reproducibility assessment; no additional reuse permission is granted by this notice.

## Security and data policy

Never commit credentials, model weights, checkpoints, raw imagery, GPS traces, voice recordings, logs, or unreviewed model outputs. See [`RELEASE_SCOPE.md`](RELEASE_SCOPE.md) for the publication-safe scope.
