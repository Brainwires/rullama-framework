#!/usr/bin/env python3
"""Convert Kokoro-82M (.pth + voicepacks) → a single GGUF the rullama engine reads.

Why GGUF: reuses the existing GgufReader + TensorFetcher + OPFS cache verbatim
(the browser OPFS cache hard-codes the GGUF magic and auto-deletes anything else).
See rullama-engine/src/reference/KOKORO_REFERENCE.md and the plan.

- Folds the (old-API) weight_norm `weight_g`/`weight_v` pairs → plain `weight`
  (no runtime weight-norm kernel needed).
- Tensor names = "k." + the PyTorch state_dict path (weight_g/v → weight).
- ALBERT's 12 layers already share one weight set in the state_dict (stored once).
- Voicepacks (voices/*.pt, [510,1,256]) → tensors "k.voice.<id>".
- Metadata: general.architecture="kokoro" + kokoro.* dims + kokoro.vocab_json.

Run in the reference venv (Intel-mac torch==2.2.2):
    ~/.cache/kokoro/venv/bin/python scripts/convert-kokoro-gguf.py \
        --dtype f32 --out ~/.cache/kokoro/kokoro-82m-f32.gguf
Use --dtype f16 for the smaller R2-shipped build.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from gguf import GGUFWriter

CACHE = os.path.expanduser("~/.cache/kokoro")


def fold_weight_norm(sd: dict) -> dict:
    """Replace every (base.weight_g, base.weight_v) pair with a folded base.weight.

    Old torch.nn.utils.weight_norm, dim=0: w = g * v / ||v|| with the L2 norm taken
    over all dims except dim 0 (per output/in-channel-0 slice). g has shape [N,1,...].
    """
    out = {}
    bases = {k[: -len(".weight_g")] for k in sd if k.endswith(".weight_g")}
    for k, v in sd.items():
        if k.endswith(".weight_g") or k.endswith(".weight_v"):
            continue
        out[k] = v
    for base in bases:
        g = sd[base + ".weight_g"].float().numpy()
        v = sd[base + ".weight_v"].float().numpy()
        axes = tuple(range(1, v.ndim))  # all dims except 0
        norm = np.sqrt((v * v).sum(axis=axes, keepdims=True))
        out[base + ".weight"] = (g * v / norm).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", default=f"{CACHE}/kokoro-v1_0.pth")
    ap.add_argument("--config", default=f"{CACHE}/config.json")
    ap.add_argument("--voices-dir", default=f"{CACHE}/voices")
    ap.add_argument("--out", default=f"{CACHE}/kokoro-82m-f32.gguf")
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    np_dtype = np.float16 if args.dtype == "f16" else np.float32

    # --- flatten {submodule: state_dict} → folded {k.<path>: ndarray} ---
    loaded = torch.load(args.pth, map_location="cpu", weights_only=True)
    tensors = {}
    for top, sd in loaded.items():
        # raw .pth keys carry a DDP "module." prefix that upstream strips at load
        flat = {
            f"{top}.{k[7:] if k.startswith('module.') else k}": val
            for k, val in sd.items()
        }
        folded = fold_weight_norm(flat)
        for name, arr in folded.items():
            a = arr if isinstance(arr, np.ndarray) else arr.float().numpy()
            tensors["k." + name] = np.ascontiguousarray(a.astype(np_dtype))

    # --- voicepacks ---
    n_voices = 0
    for vp in sorted(glob.glob(os.path.join(args.voices_dir, "*.pt"))):
        vid = os.path.splitext(os.path.basename(vp))[0]
        arr = torch.load(vp, map_location="cpu", weights_only=True).float().numpy()
        tensors[f"k.voice.{vid}"] = np.ascontiguousarray(arr.astype(np_dtype))
        n_voices += 1

    # --- write GGUF ---
    w = GGUFWriter(args.out, arch="kokoro")
    w.add_string("general.name", "kokoro-82m")
    w.add_uint32("kokoro.n_token", cfg["n_token"])
    w.add_uint32("kokoro.hidden_dim", cfg["hidden_dim"])
    w.add_uint32("kokoro.style_dim", cfg["style_dim"])
    w.add_uint32("kokoro.dim_in", cfg["dim_in"])
    w.add_uint32("kokoro.n_mels", cfg["n_mels"])
    w.add_uint32("kokoro.n_layer", cfg["n_layer"])
    w.add_uint32("kokoro.max_dur", cfg["max_dur"])
    w.add_uint32("kokoro.text_encoder_kernel_size", cfg["text_encoder_kernel_size"])
    w.add_uint32("kokoro.context_length", cfg["plbert"]["max_position_embeddings"])
    w.add_uint32("kokoro.plbert.hidden_size", cfg["plbert"]["hidden_size"])
    w.add_uint32("kokoro.plbert.num_attention_heads", cfg["plbert"]["num_attention_heads"])
    w.add_uint32("kokoro.plbert.num_hidden_layers", cfg["plbert"]["num_hidden_layers"])
    w.add_uint32("kokoro.plbert.intermediate_size", cfg["plbert"]["intermediate_size"])
    w.add_uint32("kokoro.gen_istft_n_fft", cfg["istftnet"]["gen_istft_n_fft"])
    w.add_uint32("kokoro.gen_istft_hop_size", cfg["istftnet"]["gen_istft_hop_size"])
    w.add_string("kokoro.upsample_rates_json", json.dumps(cfg["istftnet"]["upsample_rates"]))
    w.add_string("kokoro.upsample_kernel_sizes_json", json.dumps(cfg["istftnet"]["upsample_kernel_sizes"]))
    w.add_string("kokoro.resblock_kernel_sizes_json", json.dumps(cfg["istftnet"]["resblock_kernel_sizes"]))
    w.add_string("kokoro.resblock_dilation_sizes_json", json.dumps(cfg["istftnet"]["resblock_dilation_sizes"]))
    w.add_uint32("kokoro.upsample_initial_channel", cfg["istftnet"]["upsample_initial_channel"])
    w.add_string("kokoro.vocab_json", json.dumps(cfg["vocab"], ensure_ascii=False))
    w.add_uint32("kokoro.n_voices", n_voices)

    for name, arr in tensors.items():
        w.add_tensor(name, arr)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    total_bytes = sum(a.nbytes for a in tensors.values())
    print(f"wrote {args.out}")
    print(f"  tensors: {len(tensors)} ({n_voices} voices), dtype={args.dtype}, "
          f"weight bytes ~{total_bytes/1e6:.1f} MB")


if __name__ == "__main__":
    main()
