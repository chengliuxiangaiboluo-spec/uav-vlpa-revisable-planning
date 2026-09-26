# Modality-Aware Coupling with a Revisable Instruction-to-Execution Interface for Multimodal UAV Planning

This repository hosts the public project page and release materials for a multimodal UAV mission-planning framework. The framework links multimodal instruction understanding to revisable task decomposition and map-feasible planning.

## Project scope

The framework combines:

- modality-gated fusion of text, voice, gesture, and annotation inputs;
- hierarchical semantic-aware task decomposition (HSATD);
- availability-aware dynamic replanning (AADR); and
- offline visual grounding with cached predicted coordinates.

The accompanying website is available through GitHub Pages after enabling Pages for the `docs/` directory.

## Release status

This initial public release contains the project overview and a publication-safe architecture figure. Code, de-identified evaluation artifacts, protocol manifests, and scripts will be added only after a file-by-file disclosure review and confirmation of the applicable data-use permissions.

## Evidence boundary

The project evaluates separate synthetic, visual-grounding, and real-data protocols. Their scores must not be pooled. The frozen UAV-VLPA* reference is a deterministic single-run local adaptation, not an official reproduction or a five-seed training statistic.

## Security and data policy

Do not upload model weights, checkpoints, raw satellite imagery, GPS traces, real-world media, credentials, server paths, or unreviewed model-response records. See [RELEASE_SCOPE.md](RELEASE_SCOPE.md) before adding files.

## Citation

Citation metadata will be added after the author list and submission status are finalized.
