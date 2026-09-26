# Public code-release audit

## Scope checked

This release contains implementation code and strict experiment runners only. It deliberately excludes raw/controlled inputs, model weights, checkpoints, run logs, precomputed results, manuscript files, and personal machine state.

## Reproducibility boundary

The code can reconstruct the computational workflow once the authorized benchmark, split manifests, local model cache, and (for exact re-evaluation) seed checkpoints are supplied. Exact numerical reproduction cannot be claimed from this GitHub repository alone because those assets are intentionally not redistributed here.

## Mechanical gates

- Python compilation of every released source file;
- scan for previous cluster and workstation path markers;
- scan for protected model and raw-media file types; and
- no host-specific model-cache selection: deployment uses environment variables instead.

Run `python scripts/verify_release.py` before every public push.
