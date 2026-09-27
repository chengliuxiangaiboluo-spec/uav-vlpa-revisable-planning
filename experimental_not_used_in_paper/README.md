# Experimental modules not used in the paper

This directory retains development prototypes for transparency. They are **not** part of the paper-consistent execution path and were not used to produce any table, figure, statistical test, or conclusion in the associated manuscript.

`training/vlm_finetuner.py` contains a LoRA/PEFT prototype for Molmo. The reported study instead uses frozen 4-bit Molmo-7B-O inference for offline grounding. Do not invoke this directory when reproducing the reported protocols.

Install `requirements-experimental.txt` only if independently investigating these prototypes. Those dependencies are intentionally excluded from the default installation.
