#!/usr/bin/env python3
"""Dump StyleTTS2-LibriTTS hifigan Decoder reference fixtures (isolation parity).

The decoder is a deterministic function of (asr, F0_curve, N, style), so we can
parity-test the Rust port WITHOUT the upstream acoustic graph: feed fixed-seed
synthetic inputs and freeze the output audio + every stage boundary + the folded
(weight_norm-removed) weights. The HnNSF source injects random phase+noise, so we
zero `torch.rand`/`torch.randn_like` during the forward (same trick as the Kokoro
fixtures) for a reproducible reference.

This is the net-new vocoder vs Kokoro (hifigan: 4 upsamples [10,5,3,2] + direct
conv_post+tanh, NOT istftnet's iSTFT head). The Decoder class is imported straight
from the cloned repo (Modules/__init__.py is empty, so no heavy package init).

    ~/.cache/kokoro/venv/bin/python scripts/styletts2_dump_decoder_fixtures.py

Outputs → ~/.cache/styletts2/fixtures/decoder/{bin/*.bin, meta.json}
"""
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.expanduser("~/.cache/styletts2/repo")
CKPT = os.path.expanduser("~/.cache/styletts2/epochs_2nd_00020.pth")
OUT = os.path.expanduser("~/.cache/styletts2/fixtures/decoder")
SEED = 0
T_ASR = 40  # asr frames → audio = 2*T_ASR*prod([10,5,3,2]) = 24000 samples


def strip(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def main():
    os.makedirs(os.path.join(OUT, "bin"), exist_ok=True)
    sys.path.insert(0, REPO)
    from Modules.hifigan import Decoder

    dec = Decoder(
        dim_in=512, F0_channel=512, style_dim=128, dim_out=80,
        resblock_kernel_sizes=[3, 7, 11], upsample_rates=[10, 5, 3, 2],
        upsample_initial_channel=512, resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_kernel_sizes=[20, 10, 6, 4],
    ).eval()

    net = torch.load(CKPT, map_location="cpu")["net"]
    dec.load_state_dict(strip(net["decoder"]))

    # ---- fixed-seed synthetic inputs ----
    torch.manual_seed(SEED)
    asr = torch.randn(1, 512, T_ASR)
    F0_curve = torch.rand(1, 2 * T_ASR) * 200.0 + 50.0  # positive Hz, length 2T (F0_conv stride2 → T)
    N = torch.randn(1, 2 * T_ASR)
    s = torch.randn(1, 128)

    caps = {"in_asr": asr, "in_F0_curve": F0_curve, "in_N": N, "in_style": s}

    def cap(name):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            caps[name] = t
        return hook

    handles = [
        dec.encode.register_forward_hook(cap("encode")),
        dec.generator.register_forward_hook(cap("audio")),
        dec.generator.m_source.register_forward_hook(cap("har_source")),
        dec.generator.conv_post.register_forward_hook(cap("conv_post")),
    ]
    for i, blk in enumerate(dec.decode):
        blk.register_forward_hook(cap(f"decode{i}"))

    # zero the SineGen randomness so the harmonic source is reproducible
    orig_rand, orig_randn_like = torch.rand, torch.randn_like
    torch.rand = lambda *a, **k: torch.zeros(*a, dtype=torch.float32)
    torch.randn_like = lambda x, **k: torch.zeros_like(x)
    try:
        with torch.no_grad():
            audio = dec(asr, F0_curve, N, s)
    finally:
        torch.rand, torch.randn_like = orig_rand, orig_randn_like
    for h in handles:
        h.remove()
    caps["audio"] = audio

    # ---- fold weight_norm everywhere, then dump the full (named) folded state dict ----
    # (don't use the repo's Generator.remove_weight_norm — it references a nonexistent
    #  conv_pre; fold by hand over every weight_norm'd module instead.)
    for m in dec.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
    weights = {k: v for k, v in dec.state_dict().items() if "num_batches" not in k}

    shapes = {}
    for k, v in {**caps, **weights}.items():
        arr = v.detach().cpu().float().numpy() if torch.is_tensor(v) else np.asarray(v)
        arr.astype("<f4").tofile(os.path.join(OUT, "bin", f"{k}.bin"))
        shapes[k] = list(arr.shape)

    meta = {
        "T_asr": T_ASR, "n_samples": int(caps["audio"].numel()), "style_dim": 128,
        "decoder": {"upsample_rates": [10, 5, 3, 2], "upsample_kernel_sizes": [20, 10, 6, 4],
                    "resblock_kernel_sizes": [3, 7, 11], "upsample_initial_channel": 512},
        "shapes": shapes,
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("inputs/stages:", ", ".join(f"{k}{list(np.asarray(caps[k].detach().cpu()).shape)}" for k in caps))
    print(f"weights: {len(weights)} tensors; wrote fixtures → {OUT}")


if __name__ == "__main__":
    sys.exit(main())
