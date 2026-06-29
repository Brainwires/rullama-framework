#!/usr/bin/env python3
"""Convert StyleTTS2-LibriTTS (epochs_2nd_00020.pth) → a single GGUF for the rullama
voice-cloning engine. Reuses the GgufReader + OPFS cache verbatim (same as Kokoro).

Only the inference modules are shipped (alpha=beta=0 zero-shot path): style_encoder +
predictor_encoder (cloning), text_encoder, bert (PLBERT), bert_encoder, predictor,
decoder (hifigan). Diffusion, text_aligner, pitch_extractor and the GAN discriminators
are dropped. weight_norm and spectral_norm are folded offline by INSTANTIATING the
modules and calling remove_* (the exact, validated folding used by the parity fixtures),
so the runtime needs no norm kernels.

Tensor names = "<module>." + folded PyTorch state-dict path (matches the Rust loader).

Run in the reference venv (needs: munch einops einops_exts transformers torchaudio):
    ~/.cache/kokoro/venv/bin/python scripts/convert-styletts2-gguf.py --dtype f16
"""
import argparse
import os
import sys

import numpy as np
import torch
from gguf import GGUFWriter

REPO = os.path.expanduser("~/.cache/styletts2/repo")
CKPT = os.path.expanduser("~/.cache/styletts2/epochs_2nd_00020.pth")
sys.path.insert(0, REPO)


def strip(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def fold(module):
    """remove_weight_norm + remove_spectral_norm in place (validated folding)."""
    for m in module.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
        if hasattr(m, "weight_orig"):
            torch.nn.utils.remove_spectral_norm(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f16")
    args = ap.parse_args()
    out = args.out or os.path.expanduser(f"~/.cache/styletts2/styletts2-libritts-{args.dtype}.gguf")
    np_dtype = np.float16 if args.dtype == "f16" else np.float32

    import models
    from Modules.hifigan import Decoder
    from Utils.PLBERT.util import load_plbert

    net = torch.load(args.ckpt, map_location="cpu")["net"]
    mods = {}
    te = models.TextEncoder(channels=512, kernel_size=5, depth=3, n_symbols=178).eval()
    te.load_state_dict(strip(net["text_encoder"]))
    mods["text_encoder"] = te
    pp = models.ProsodyPredictor(style_dim=128, d_hid=512, nlayers=3, max_dur=50, dropout=0.2).eval()
    pp.load_state_dict(strip(net["predictor"]))
    mods["predictor"] = pp
    be = torch.nn.Linear(768, 512).eval()
    be.load_state_dict(strip(net["bert_encoder"]))
    mods["bert_encoder"] = be
    bert = load_plbert(os.path.join(REPO, "Utils/PLBERT")).eval()
    bert.load_state_dict(strip(net["bert"]), strict=False)
    mods["bert"] = bert
    for key in ("style_encoder", "predictor_encoder"):
        se = models.StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512).eval()
        se.load_state_dict(strip(net[key]))
        mods[key] = se
    dec = Decoder(dim_in=512, F0_channel=512, style_dim=128, dim_out=80,
                  resblock_kernel_sizes=[3, 7, 11], upsample_rates=[10, 5, 3, 2],
                  upsample_initial_channel=512, resblock_dilation_sizes=[[1, 3, 5]] * 3,
                  upsample_kernel_sizes=[20, 10, 6, 4]).eval()
    dec.load_state_dict(strip(net["decoder"]))
    mods["decoder"] = dec

    # Style-diffusion denoiser (StyleTransformer1d) — restores natural prosody via the
    # alpha=0.3/beta=0.7 sampler. channels=style_dim*2=256, context_embedding=PLBERT(768),
    # context_features=256. The checkpoint stores it under module.diffusion.net.* (with a
    # duplicate module.diffusion.unet.* we skip). No weight/spectral norm → no folding.
    from Modules.diffusion.modules import StyleTransformer1d
    diff = StyleTransformer1d(channels=256, context_embedding_features=768, context_features=256,
                              num_layers=3, num_heads=8, head_features=64, multiplier=2).eval()
    dnet = {k[len("module.diffusion.net."):]: v for k, v in net["diffusion"].items()
            if k.startswith("module.diffusion.net.")}
    diff.load_state_dict(dnet)
    mods["diffusion"] = diff

    # sigma_data for the KDiffusion denoise_fn scale weights (config dist.sigma_data; the
    # placeholder is what public inference uses since it isn't persisted in the checkpoint).
    sigma_data = 0.2

    for m in mods.values():
        fold(m)

    w = GGUFWriter(out, "styletts2")  # 2nd arg is general.architecture
    w.add_string("general.name", "StyleTTS2-LibriTTS")
    w.add_uint32("styletts2.n_token", 178)
    w.add_uint32("styletts2.hidden_dim", 512)
    w.add_uint32("styletts2.style_dim", 128)
    w.add_uint32("styletts2.n_layer", 3)
    w.add_uint32("styletts2.max_dur", 50)
    w.add_uint32("styletts2.plbert_hidden", 768)
    w.add_uint32("styletts2.plbert_heads", 12)
    w.add_uint32("styletts2.plbert_layers", 12)
    w.add_uint32("styletts2.plbert_inter", 2048)
    w.add_uint32("styletts2.plbert_emb", 128)
    w.add_array("styletts2.upsample_rates", [10, 5, 3, 2])
    w.add_array("styletts2.upsample_kernel_sizes", [20, 10, 6, 4])
    w.add_array("styletts2.resblock_kernel_sizes", [3, 7, 11])
    # mel frontend (compute_style): torchaudio default sr=16000 quirk
    w.add_uint32("styletts2.mel_n_fft", 2048)
    w.add_uint32("styletts2.mel_hop", 300)
    w.add_uint32("styletts2.mel_win", 1200)
    w.add_uint32("styletts2.mel_n_mels", 80)
    w.add_uint32("styletts2.mel_sr", 16000)
    # style-diffusion sampler params (ADPM2 + Karras schedule, empirical in the LibriTTS demo)
    w.add_float32("styletts2.diff_sigma_data", sigma_data)
    w.add_float32("styletts2.diff_sigma_min", 1e-4)
    w.add_float32("styletts2.diff_sigma_max", 3.0)
    w.add_float32("styletts2.diff_rho", 9.0)
    w.add_uint32("styletts2.diff_steps", 5)

    n = 0
    for pfx, mod in mods.items():
        for k, v in mod.state_dict().items():
            if "num_batches_tracked" in k or k.endswith("position_ids"):
                continue
            arr = v.detach().cpu().float().numpy().astype(np_dtype)
            w.add_tensor(f"{pfx}.{k}", arr)
            n += 1

    # bake the compute_style mel frontend (torchaudio Slaney filterbank + Hann window;
    # the default-sr=16000 quirk) so the Rust encoder needs no mel construction.
    import torchaudio
    to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    w.add_tensor("mel.filterbank", to_mel.mel_scale.fb.detach().numpy().astype(np.float32))  # [1025,80]
    w.add_tensor("mel.window", to_mel.spectrogram.window.detach().numpy().astype(np.float32))  # [1200]
    n += 2

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    sz = os.path.getsize(out)
    print(f"wrote {n} tensors → {out} ({sz/1e6:.1f} MB, {args.dtype})")


if __name__ == "__main__":
    sys.exit(main())
