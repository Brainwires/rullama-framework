#!/usr/bin/env python3
"""Freeze style-diffusion parity fixtures for the Rust oracle.

Isolates the diffusion sampler: dumps PyTorch's bert_dur (PLBERT output) + ref_s + the
exact replayed noise sequence as *inputs*, and s_pred as the expected output. The Rust
oracle feeds the same bert_dur/ref_s and replays the same noise, so any mismatch is the
denoiser/sampler math, not PLBERT or the RNG.

  ~/.cache/kokoro/venv/bin/python scripts/styletts2_dump_diffusion_fixtures.py

Writes <home>/.cache/styletts2/fixtures/diffusion/*.bin (f32 little-endian) + meta.json.
"""
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.expanduser("~/.cache/styletts2/repo")
CKPT = os.path.expanduser("~/.cache/styletts2/epochs_2nd_00020.pth")
OUT = os.path.expanduser("~/.cache/styletts2/fixtures/diffusion")
sys.path.insert(0, REPO)

SIGMA_DATA, SIGMA_MIN, SIGMA_MAX, RHO, STEPS = 0.2, 1e-4, 3.0, 9.0, 5


def w(name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(os.path.join(OUT, f"{name}.bin"))
    return list(arr.shape)


def main():
    os.makedirs(OUT, exist_ok=True)
    import models
    from Modules.diffusion.modules import StyleTransformer1d
    from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
    from Utils.PLBERT.util import load_plbert

    net = torch.load(CKPT, map_location="cpu")["net"]
    strip = lambda sd, p: {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}

    bert = load_plbert(os.path.join(REPO, "Utils/PLBERT")).eval()
    bert.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in net["bert"].items()}, strict=False)
    diff = StyleTransformer1d(channels=256, context_embedding_features=768, context_features=256,
                              num_layers=3, num_heads=8, head_features=64, multiplier=2).eval()
    diff.load_state_dict(strip(net["diffusion"], "module.diffusion.net."))
    # dump the denoiser weights too, so the Rust parity test is self-contained (no GGUF).
    for k, v in diff.state_dict().items():
        if k.startswith("fixed_embedding"):
            continue  # unused at embedding_scale=1
        w(f"diffusion.{k}", v.detach().numpy())

    # KDiffusion wrapper just for denoise_fn (sigma_data scale weights)
    from Modules.diffusion.sampler import KDiffusion, LogNormalDistribution
    kdiff = KDiffusion(net=diff, sigma_distribution=LogNormalDistribution(-3.0, 1.0), sigma_data=SIGMA_DATA)
    sampler = DiffusionSampler(kdiff, sampler=ADPM2Sampler(),
                               sigma_schedule=KarrasSchedule(SIGMA_MIN, SIGMA_MAX, RHO), clamp=False)

    # fixed tokens (avoid espeak); a plausible phoneme-id sequence
    torch.manual_seed(1234)
    ids = torch.randint(1, 178, (1, 28))
    ids[0, 0] = 0  # BOS
    input_lengths = torch.LongTensor([ids.shape[-1]])
    text_mask = torch.arange(ids.shape[-1])[None, :] >= input_lengths[:, None]  # length_to_mask
    with torch.no_grad():
        bert_dur = bert(ids, attention_mask=(~text_mask).int())  # [1, L, 768]
    L = bert_dur.shape[1]

    # reference style vector (seeded; parity only needs both sides identical)
    ref_s = torch.randn(1, 256, generator=torch.Generator().manual_seed(7))

    # pre-generate + capture the exact noise the sampler will consume: initial + 4 step noises
    g = torch.Generator().manual_seed(42)
    noise_init = torch.randn(1, 256, generator=g).unsqueeze(1)  # [1,1,256]
    step_noises = [torch.randn(1, 1, 256, generator=g) for _ in range(STEPS - 1)]
    queue = list(step_noises)
    orig_randn_like = torch.randn_like
    torch.randn_like = lambda x, *a, **k: queue.pop(0)

    with torch.no_grad():
        s_pred = sampler(noise=noise_init, embedding=bert_dur, embedding_scale=1.0,
                         features=ref_s, num_steps=STEPS).squeeze(1)  # [1, 256]
    torch.randn_like = orig_randn_like

    # one isolated denoiser eval for first-fault debugging: net(c_in*x, c_noise, emb, feat)
    sig0 = torch.tensor([SIGMA_MAX])
    c_in = (sig0 ** 2 + SIGMA_DATA ** 2) ** -0.5
    c_noise = torch.log(sig0) * 0.25
    x0 = sig0 * noise_init  # sampler's x = sigmas[0]*noise
    with torch.no_grad():
        net_out = diff(c_in * x0, c_noise, embedding=bert_dur, features=ref_s, embedding_scale=1.0)  # [1,1,256]

    meta = {
        "L": int(L), "sigma_data": SIGMA_DATA, "sigma_min": SIGMA_MIN, "sigma_max": SIGMA_MAX,
        "rho": RHO, "steps": STEPS,
        "shapes": {
            "bert_dur": w("bert_dur", bert_dur.numpy()),
            "ref_s": w("ref_s", ref_s.numpy()),
            "noise_init": w("noise_init", noise_init.numpy()),
            "step_noises": w("step_noises", torch.cat(step_noises, 0).numpy()),  # [4,1,256]
            "net_out": w("net_out", net_out.numpy()),
            "s_pred": w("s_pred", s_pred.numpy()),
        },
    }
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(f"dumped diffusion fixtures (L={L}) → {OUT}")
    print("s_pred[:6] =", s_pred[0, :6].tolist())


if __name__ == "__main__":
    main()
