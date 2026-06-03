#!/usr/bin/env python3
"""Dump StyleTTS2-LibriTTS style-encoder reference fixtures for the Rust CPU oracle.

The cloning capability Kokoro dropped lives in StyleTTS2's two `StyleEncoder`s
(acoustic `style_encoder` + prosodic `predictor_encoder`). This runs them on a
reference clip and freezes:
  - the input mel (so the Rust encoder oracle is tested in isolation),
  - every ResBlk-boundary tensor,
  - the final 128-d acoustic + 128-d prosodic + 256-d concat style vector,
  - the spectral-norm-FOLDED conv/linear weights (named, little-endian f32 .bin),
  - mel-frontend fixtures (audio slice + torchaudio Slaney filterbank + window +
    expected mel) so the Rust mel can be validated separately.

The StyleEncoder/ResBlk/DownSample classes are copied verbatim from
`yl4579/StyleTTS2@main models.py` (lines 27-164) so the dump needs none of the
repo's heavy deps (PLBERT, monotonic_align, ...).

Run in the pinned reference venv (Intel-mac → torch==2.2.2, py3.12):
    ~/.cache/kokoro/venv/bin/python scripts/styletts2_dump_style_fixtures.py

Outputs → ~/.cache/styletts2/fixtures/{tensors.npz, bin/*.bin, meta.json}
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_spectral_norm, spectral_norm

CKPT = os.path.expanduser("~/.cache/styletts2/epochs_2nd_00020.pth")
REF_WAV = os.path.expanduser("~/.cache/kokoro/tts_demo.wav")  # any 24 kHz mono clip
OUT = os.path.expanduser("~/.cache/styletts2/fixtures")

# ---- verbatim from StyleTTS2 models.py (the parts the encoder needs) ----


class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "half":
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        if self.layer_type == "half":
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        raise RuntimeError(self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample="none"):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def _residual(self, x):
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=64, style_dim=128, max_conv_dim=512):
        super().__init__()
        blocks = [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]
        for _ in range(4):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample="half")]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        return self.unshared(h)


# ---- mel frontend, exactly as StyleTTS2 compute_style (NOTE: torchaudio default
#      sample_rate=16000 is used even though audio is 24 kHz — replicate verbatim) ----
import torchaudio

to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
MEAN, STD = -4.0, 4.0


def preprocess(wave):
    mel = to_mel(torch.from_numpy(wave).float())
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - MEAN) / STD
    return mel  # [1, 80, T]


def strip(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def fold_weights(enc, prefix, out):
    """remove_spectral_norm everywhere, then dump folded conv/linear weights by name."""
    for m in enc.modules():
        if hasattr(m, "weight_orig"):
            remove_spectral_norm(m)
    s = enc.shared
    conv0 = s[0]
    out[f"{prefix}.conv0.weight"] = conv0.weight.detach().numpy()
    out[f"{prefix}.conv0.bias"] = conv0.bias.detach().numpy()
    for i in range(4):
        blk = s[1 + i]
        out[f"{prefix}.blk{i}.conv1.weight"] = blk.conv1.weight.detach().numpy()
        out[f"{prefix}.blk{i}.conv1.bias"] = blk.conv1.bias.detach().numpy()
        out[f"{prefix}.blk{i}.conv2.weight"] = blk.conv2.weight.detach().numpy()
        out[f"{prefix}.blk{i}.conv2.bias"] = blk.conv2.bias.detach().numpy()
        out[f"{prefix}.blk{i}.down.weight"] = blk.downsample_res.conv.weight.detach().numpy()
        out[f"{prefix}.blk{i}.down.bias"] = blk.downsample_res.conv.bias.detach().numpy()
        if blk.learned_sc:
            out[f"{prefix}.blk{i}.sc.weight"] = blk.conv1x1.weight.detach().numpy()
    conv_out = s[6]
    out[f"{prefix}.conv_out.weight"] = conv_out.weight.detach().numpy()
    out[f"{prefix}.conv_out.bias"] = conv_out.bias.detach().numpy()
    out[f"{prefix}.linear.weight"] = enc.unshared.weight.detach().numpy()
    out[f"{prefix}.linear.bias"] = enc.unshared.bias.detach().numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    import soundfile as sf

    wave, sr = sf.read(REF_WAV)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != 24000:
        wave = torchaudio.functional.resample(torch.from_numpy(wave).float(), sr, 24000).numpy()
    wave = wave.astype(np.float32)

    mel = preprocess(wave)  # [1, 80, T]
    enc_in = mel.unsqueeze(1)  # [1, 1, 80, T]

    params = torch.load(CKPT, map_location="cpu")
    net = params["net"] if "net" in params else params
    print("checkpoint net keys:", list(net.keys())[:20])

    caps = {}
    caps["mel"] = mel.detach().numpy()  # encoder input (after unsqueeze(1) in the oracle)

    for prefix, key in [("acoustic", "style_encoder"), ("prosodic", "predictor_encoder")]:
        enc = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512).eval()
        enc.load_state_dict(strip(net[key]))

        # capture per-ResBlk outputs (spectral_norm still active = correct weights)
        blk_caps = []
        handles = [enc.shared[1 + i].register_forward_hook(
            lambda _m, _i, o, idx=i: blk_caps.append((idx, o.detach().numpy()))) for i in range(4)]
        with torch.no_grad():
            s = enc(enc_in)
        for h in handles:
            h.remove()
        caps[f"{prefix}.style"] = s.detach().numpy()  # [1, 128]
        for idx, t in blk_caps:
            caps[f"{prefix}.blk{idx}.out"] = t

        fold_weights(enc, prefix, caps)  # remove_spectral_norm + dump folded weights

    caps["concat256"] = np.concatenate([caps["acoustic.style"], caps["prosodic.style"]], axis=1)

    # mel-frontend fixtures (validate the Rust mel separately)
    caps["audio"] = wave
    caps["mel_filterbank"] = to_mel.mel_scale.fb.detach().numpy()  # [n_freqs=1025, n_mels=80]
    caps["mel_window"] = to_mel.spectrogram.window.detach().numpy()  # Hann(win_length=1200)

    np.savez(os.path.join(OUT, "tensors.npz"), **caps)
    bindir = os.path.join(OUT, "bin")
    os.makedirs(bindir, exist_ok=True)
    for k, v in caps.items():
        np.asarray(v).astype("<f4").tofile(os.path.join(bindir, f"{k}.bin"))

    meta = {
        "ref_wav": REF_WAV, "sample_rate_audio": 24000,
        "mel": {"sample_rate": 16000, "n_fft": 2048, "win_length": 1200, "hop_length": 300,
                "n_mels": 80, "log_eps": 1e-5, "mean": MEAN, "std": STD, "trim": "skipped (parity)"},
        "style_dim": 128, "full_ref_s": 256,
        "shapes": {k: list(np.asarray(v).shape) for k, v in caps.items()},
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("captured:", ", ".join(f"{k}{list(np.asarray(v).shape)}" for k, v in caps.items()))
    print(f"wrote fixtures → {OUT}")


if __name__ == "__main__":
    sys.exit(main())
