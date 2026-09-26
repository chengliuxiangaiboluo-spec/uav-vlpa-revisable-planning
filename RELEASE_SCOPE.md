# Public-release scope

## Allowed after review

- Source code that contains no credentials, local absolute paths, or private URLs.
- Sanitized configuration files and evaluation entrypoints.
- Final figures created by the authors.
- De-identified aggregate summaries, protocol matrices, split identifiers, and planned-contrast tables.
- Documentation that distinguishes protocol-specific results.

## Excluded

- Model weights, checkpoints, tokenizer snapshots, or licensed third-party model files.
- Raw satellite images, map tiles, GPS traces, voice, gesture, annotation media, or participant-related records.
- Server logs, Slurm files, hostnames, usernames, absolute paths, credentials, access tokens, and environment secrets.
- Raw model outputs unless they are reviewed for permitted redistribution and the documentation explicitly states their inclusion.

## Required before adding reproducibility artifacts

1. Confirm the relevant data-use and redistribution permissions.
2. Remove private paths, user identifiers, credentials, and raw restricted content.
3. Regenerate SHA-256 inventories after the final public file set is frozen.
4. Add a licence only after all authors and the institution approve its terms.
