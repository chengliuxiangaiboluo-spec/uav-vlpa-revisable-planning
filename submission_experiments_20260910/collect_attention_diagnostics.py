"""Extract actual cross-modal attention weights from one trained seed.

Only held-out scenarios with readable voice, gesture, and annotation inputs
are admitted.  The hook captures the attention returned by the installed
``torch.nn.MultiheadAttention`` modules; it does not infer weights from labels,
inject values, or use planning metrics.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.experiment_config import get_default_config
from submission_common import load_checkpoint, read_samples, run_metadata, write_json
from train_submission_seed import build_fuser
from utils.offline_config import setup_offline_environment
from utils.seed_manager import set_global_seed

MODALITIES = ("Text", "Voice", "Gesture", "Annotation")


def _load_audio(path: str, device: str | torch.device) -> torch.Tensor | None:
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        waveform = waveform[:1, :48000]
        if waveform.shape[1] < 48000:
            waveform = torch.nn.functional.pad(waveform, (0, 48000 - waveform.shape[1]))
        return waveform.to(device)
    except Exception:
        return None


def _load_image(path: str, device: str | torch.device) -> torch.Tensor | None:
    try:
        from PIL import Image
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    except Exception:
        return None


def _eligible(sample) -> bool:
    fields = (sample.audio_path, sample.gesture_image_path, sample.annotation_image_path)
    return bool(sample.text_instruction) and all(path and os.path.isfile(path) for path in fields)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect one seed of held-out cross-modal attention")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, help="Smoke-test limit only")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()
    if args.max_samples is not None and (not args.smoke or args.max_samples < 1):
        raise ValueError("--max-samples is allowed only for a positive --smoke run")

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite attention worker output: {output}")
    setup_offline_environment(server_mode=args.server)
    set_global_seed(args.seed)
    base_cfg, _, model_cfg, _, _ = get_default_config()
    device = base_cfg.device
    run_root = Path(args.run_root).resolve()
    checkpoint_dir = run_root / "models" / "ours" / f"seed_{args.seed}"
    fuser = build_fuser("ours", model_cfg, device)
    audio_encoder = getattr(fuser, "audio_encoder", None)
    if audio_encoder is not None and hasattr(audio_encoder, "_load_model"):
        audio_encoder._load_model()
    checkpoint = load_checkpoint(fuser, checkpoint_dir, "best_fusion_model", device)
    fuser.eval()

    # ``build_fuser`` resolves ``model_cfg.text_model`` to the project's local
    # Weights/all-MiniLM-L6-v2 directory.  Reuse this exact encoder rather than
    # constructing a second encoder from the unresolved Hugging Face identifier;
    # the latter would incorrectly try the network on an offline node.
    text_encoder = fuser.text_encoder
    text_encoder._load_model()
    if getattr(text_encoder, "_sbert", None) == "fallback":
        raise RuntimeError(
            "The fuser text encoder fell back to random embeddings; refusing "
            "attention extraction. Check the local all-MiniLM-L6-v2 directory."
        )
    print(f"[Attention] text encoder loaded from {text_encoder.model_name}", flush=True)

    all_test = read_samples(args.test_file)
    candidates = [sample for sample in all_test if _eligible(sample)]
    if args.max_samples:
        candidates = candidates[:args.max_samples]
    if not candidates:
        raise RuntimeError("No held-out samples have all three readable non-text modalities")

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(fuser.cross_attention.layers):
        def hook(_module, _inputs, output, index=layer_index):
            weights = output[1]
            if weights is None:
                raise RuntimeError("MultiheadAttention did not return attention weights")
            captured[index] = weights.detach().cpu()
        handles.append(layer.attn.register_forward_hook(hook))

    sums = np.zeros((len(fuser.cross_attention.layers), len(MODALITIES)), dtype=np.float64)
    used_ids, skipped_ids = [], []
    try:
        with torch.no_grad():
            for position, sample in enumerate(candidates, start=1):
                captured.clear()
                audio = _load_audio(sample.audio_path, device)
                gesture = _load_image(sample.gesture_image_path, device)
                annotation = _load_image(sample.annotation_image_path, device)
                if audio is None or gesture is None or annotation is None:
                    skipped_ids.append(str(sample.scenario_id))
                    continue
                text = text_encoder.encode_texts([sample.text_instruction]).to(device)
                fuser(text_emb=text, audio_values=audio, gesture_images=gesture,
                      annotation_images=annotation, return_bias=False)
                if set(captured) != set(range(len(fuser.cross_attention.layers))):
                    raise RuntimeError("Attention hook did not capture every cross-modal layer")
                for layer_index, weights in captured.items():
                    flat = weights.squeeze(0).squeeze(0).numpy()
                    if flat.shape != (len(MODALITIES),):
                        raise RuntimeError(f"Unexpected attention shape {flat.shape}; expected {len(MODALITIES)} modalities")
                    sums[layer_index] += flat
                used_ids.append(str(sample.scenario_id))
                if position % 10 == 0 or position == len(candidates):
                    print(f"[Attention] seed={args.seed} | processed={position}/{len(candidates)} | usable={len(used_ids)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    if not used_ids:
        raise RuntimeError("No sample survived tensor loading; no attention result is valid")
    means = sums / len(used_ids)
    row_sums = means.sum(axis=1)
    if not np.allclose(row_sums, np.ones_like(row_sums), atol=1e-5):
        raise RuntimeError(f"Attention rows do not sum to one: {row_sums}")

    output.mkdir(parents=True)
    write_json(output / "attention_seed.json", run_metadata({
        "experiment": "Held-out cross-modal attention extraction",
        "seed": args.seed,
        "test_file": str(Path(args.test_file).resolve()),
        "fusion_checkpoint": str(checkpoint),
        "modalities": list(MODALITIES),
        "attention_semantics": "text-query attention over the four assembled modality tokens; not causal importance",
        "candidate_count": len(candidates),
        "usable_count": len(used_ids),
        "used_scenario_ids": used_ids,
        "skipped_scenario_ids": skipped_ids,
        "layer_mean_attention": means.tolist(),
        "layer_row_sums": row_sums.tolist(),
        "smoke_test": bool(args.smoke),
    }))
    print(f"[Attention] complete | seed={args.seed} | usable={len(used_ids)} | output={output}", flush=True)


if __name__ == "__main__":
    main()
