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

    torch.manual_seed(SEED)  # re-seed right before forward so source noise is reproducible
    with torch.no_grad():
        out = model(phonemes, ref_s, speed=1, return_output=True)
    audio = out.audio.cpu().float().numpy()
    pred_dur = out.pred_dur.cpu().numpy()
    caps["audio"] = audio
    caps["pred_dur"] = pred_dur.astype(np.int64)
    caps["ref_s"] = ref_s.cpu().float().numpy()
    caps["input_ids"] = np.array(input_ids, dtype=np.int64)

    np.savez(os.path.join(OUT, "tensors.npz"), **caps)

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
