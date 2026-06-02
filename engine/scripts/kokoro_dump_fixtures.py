#!/usr/bin/env python3
"""Dump Kokoro-82M reference fixtures for the Rust CPU-oracle parity tests.

Runs the upstream `hexgrad/kokoro` KModel on a fixed phrase + voice with a fixed
RNG seed (the ISTFTNet source module is otherwise non-deterministic), and saves
every stage-boundary tensor + the final 24 kHz WAV. The Rust oracle in
`crates/rullama/src/reference/` diffs against these (see KOKORO_REFERENCE.md).

Run inside the pinned reference venv (Intel-mac → torch==2.2.2, py3.12):
    ~/.cache/kokoro/venv/bin/python scripts/kokoro_dump_fixtures.py

Outputs → ~/.cache/kokoro/fixtures/{tensors.npz, meta.json, ref.wav}
"""
import json
import os
import sys

import numpy as np
import torch

CACHE = os.path.expanduser("~/.cache/kokoro")
OUT = os.path.join(CACHE, "fixtures")
SEED = 0
TEXT = "Hello, how are you today?"
VOICE = "af_heart"


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)

    from kokoro import KModel
    from misaki import en

    # --- G2P (also a fixture: validate the Rust G2P against this phoneme string) ---
    g2p = en.G2P(trf=False, british=False)
    phonemes, _ = g2p(TEXT)
    print(f"text: {TEXT!r}\nphonemes: {phonemes!r}")

    # --- model (load strictly from the local cache, never hit HF) ---
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=os.path.join(CACHE, "config.json"),
        model=os.path.join(CACHE, "kokoro-v1_0.pth"),
        disable_complex=False,  # exact TorchSTFT path — what we port (see KOKORO_REFERENCE.md)
    ).eval()

    # --- voice vector: voicepack row selected by token length ---
    vp = torch.load(os.path.join(CACHE, "voices", f"{VOICE}.pt"), map_location="cpu")
    input_ids = [0, *[model.vocab.get(p) for p in phonemes if model.vocab.get(p) is not None], 0]
    ref_s = vp[len(input_ids) - 1]  # [1, 256]

    caps = {}

    def cap(name):
        def hook(_m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                caps[name] = t.detach().cpu().float().numpy()
        return hook

    # stage-boundary hooks (names mirror KOKORO_REFERENCE.md dataflow)
    model.bert.register_forward_hook(cap("bert"))
    model.bert_encoder.register_forward_hook(cap("bert_encoder"))
    model.predictor.text_encoder.register_forward_hook(cap("pred_text_encoder_d"))
    model.predictor.duration_proj.register_forward_hook(cap("duration_logits"))
    model.predictor.shared.register_forward_hook(cap("pred_shared"))
    model.predictor.F0_proj.register_forward_hook(cap("F0"))
    model.predictor.N_proj.register_forward_hook(cap("N"))
    model.text_encoder.register_forward_hook(cap("text_encoder_ten"))
    model.decoder.encode.register_forward_hook(cap("dec_encode"))
    model.decoder.generator.register_forward_hook(cap("dec_generator"))

    # generator-isolation fixtures: its input x, and the harmonic source `har`
    # (har is non-deterministic — random phase+noise — so inject it for parity).
    def cap_pre(name, idx=0):
        def hook(_m, inp):
            caps[name] = inp[idx].detach().cpu().float().numpy()
        return hook

    h_gx = model.decoder.generator.register_forward_pre_hook(cap_pre("gen_x", 0))
    h_gh = model.decoder.generator.noise_convs[0].register_forward_pre_hook(cap_pre("gen_har", 0))

    torch.manual_seed(SEED)  # re-seed right before forward so source noise is reproducible
    with torch.no_grad():
        out = model(phonemes, ref_s, speed=1, return_output=True)
    audio = out.audio.cpu().float().numpy()
    pred_dur = out.pred_dur.cpu().numpy()
    caps["audio"] = audio
    caps["pred_dur"] = pred_dur.astype(np.int64)
    caps["ref_s"] = ref_s.cpu().float().numpy()
    caps["input_ids"] = np.array(input_ids, dtype=np.int64)

    # ---- deterministic pass: zero the SineGen randomness (rand_ini + noise) so the
    # harmonic source is reproducible, for validating the standalone Rust source port.
    h_gx.remove()  # stop the seeded gen_x/gen_har hooks from clobbering on the 2nd forward
    h_gh.remove()
    cap_har_det = []
    h_det = model.decoder.generator.noise_convs[0].register_forward_pre_hook(
        lambda _m, inp: cap_har_det.append(inp[0].detach().cpu().float().numpy())
    )
    cap_src = []
    h_src = model.decoder.generator.m_source.register_forward_hook(
        lambda _m, _i, out: cap_src.append(out[0].squeeze().detach().cpu().float().numpy())
    )
    orig_rand, orig_randn_like = torch.rand, torch.randn_like
    torch.rand = lambda *s, **k: torch.zeros(*s, dtype=torch.float32)
    torch.randn_like = lambda x, **k: torch.zeros_like(x)
    try:
        with torch.no_grad():
            out_det = model(phonemes, ref_s, speed=1, return_output=True)
    finally:
        torch.rand, torch.randn_like = orig_rand, orig_randn_like
        h_det.remove()
        h_src.remove()
    caps["audio_det"] = out_det.audio.cpu().float().numpy()
    caps["gen_har_det"] = cap_har_det[-1]
    caps["har_source_det"] = cap_src[-1]

    np.savez(os.path.join(OUT, "tensors.npz"), **caps)

    # also dump raw little-endian f32 .bin per tensor so the Rust oracle's parity
    # tests can read fixtures self-contained (no npz/zip parsing in Rust).
    bindir = os.path.join(OUT, "bin")
    os.makedirs(bindir, exist_ok=True)
    for k, v in caps.items():
        v.astype("<f4").tofile(os.path.join(bindir, f"{k}.bin"))

    # 24 kHz mono WAV
    try:
        import soundfile as sf
        sf.write(os.path.join(OUT, "ref.wav"), audio, 24000)
    except Exception as e:  # noqa: BLE001
        print(f"(soundfile unavailable, skipped wav: {e})", file=sys.stderr)

    meta = {
        "text": TEXT, "voice": VOICE, "seed": SEED, "phonemes": phonemes,
        "input_ids": input_ids, "n_phonemes": len(input_ids),
        "pred_dur": pred_dur.tolist(), "n_samples": int(audio.shape[-1]),
        "shapes": {k: list(v.shape) for k, v in caps.items()},
        "disable_complex": False, "sample_rate": 24000,
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("captured stages:", ", ".join(f"{k}{list(v.shape)}" for k, v in caps.items()))
    print(f"wrote fixtures → {OUT}")


if __name__ == "__main__":
    main()
