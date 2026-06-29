#!/usr/bin/env python3
"""Full StyleTTS2-LibriTTS zero-shot synthesis reference (alpha=beta=0, hifigan).

Clones a target clip's voice and synthesizes text in it — the whole pipeline:
  compute_style(ref_wav) -> [acoustic128 ‖ prosodic128]
  text_encoder + bert(PLBERT) + bert_encoder + predictor(dur,F0,N) + decoder
Dumps the reference cloned WAV + every stage tensor + the tokens, so the Rust
acoustic-graph port (text_encoder/bert/bert_encoder/predictor — mostly mirrors the
Kokoro reference) can be validated end to end. Decoder + style encoder are already
ported and validated; this provides the upstream targets and the final audio.

Modules are imported straight from the cloned repo (models.py now importable after
`pip install munch einops einops_exts`); we instantiate TextEncoder/ProsodyPredictor/
StyleEncoder/Decoder directly and load per-module weights from the checkpoint, skipping
build_model (which would pull diffusion + WavLM discriminators we don't use).

Target voice = ~/.cache/kokoro/tts_demo.wav (the Kokoro voice we already have).

    ~/.cache/kokoro/venv/bin/python scripts/styletts2_dump_synth_fixtures.py

Outputs → ~/.cache/styletts2/fixtures/synth/{bin/*.bin, meta.json, cloned.wav}
"""
import json
import os
import sys

import numpy as np
import torch
import torchaudio

REPO = os.path.expanduser("~/.cache/styletts2/repo")
CKPT = os.path.expanduser("~/.cache/styletts2/epochs_2nd_00020.pth")
REF_WAV = os.path.expanduser("~/.cache/kokoro/tts_demo.wav")
OUT = os.path.expanduser("~/.cache/styletts2/fixtures/synth")
TEXT = "Hello from rullama, this is my cloned voice."

sys.path.insert(0, REPO)


def strip(sd):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)


def compute_style(path, style_enc, pred_enc):
    import soundfile as sf
    wave, sr = sf.read(path)
    if wave.ndim > 1:
        wave = wave.mean(1)
    if sr != 24000:
        wave = torchaudio.functional.resample(torch.from_numpy(wave).float(), sr, 24000).numpy()
    mel = to_mel(torch.from_numpy(wave).float())
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - (-4)) / 4
    mel = mel.unsqueeze(1)
    with torch.no_grad():
        a = style_enc(mel)
        p = pred_enc(mel)
    return torch.cat([a, p], dim=1)  # [1, 256]


def main():
    os.makedirs(os.path.join(OUT, "bin"), exist_ok=True)
    import models
    from Modules.hifigan import Decoder
    from Utils.PLBERT.util import load_plbert
    from text_utils import TextCleaner

    net = torch.load(CKPT, map_location="cpu")["net"]

    text_encoder = models.TextEncoder(channels=512, kernel_size=5, depth=3, n_symbols=178).eval()
    text_encoder.load_state_dict(strip(net["text_encoder"]))
    predictor = models.ProsodyPredictor(style_dim=128, d_hid=512, nlayers=3, max_dur=50, dropout=0.2).eval()
    predictor.load_state_dict(strip(net["predictor"]))
    bert_encoder = torch.nn.Linear(768, 512).eval()
    bert_encoder.load_state_dict(strip(net["bert_encoder"]))
    bert = load_plbert(os.path.join(REPO, "Utils/PLBERT")).eval()
    bert.load_state_dict(strip(net["bert"]), strict=False)
    style_enc = models.StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512).eval()
    style_enc.load_state_dict(strip(net["style_encoder"]))
    pred_enc = models.StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512).eval()
    pred_enc.load_state_dict(strip(net["predictor_encoder"]))
    decoder = Decoder(dim_in=512, F0_channel=512, style_dim=128, dim_out=80,
                      resblock_kernel_sizes=[3, 7, 11], upsample_rates=[10, 5, 3, 2],
                      upsample_initial_channel=512, resblock_dilation_sizes=[[1, 3, 5]] * 3,
                      upsample_kernel_sizes=[20, 10, 6, 4]).eval()
    decoder.load_state_dict(strip(net["decoder"]))

    # ---- style from the target clip ----
    ref_s = compute_style(REF_WAV, style_enc, pred_enc)  # [1,256]
    s = ref_s[:, 128:]   # prosodic
    ref = ref_s[:, :128] # acoustic

    # ---- phonemes -> tokens (misaki, skipping espeak) ----
    from misaki import en
    g2p = en.G2P(trf=False, british=False)
    ps, _ = g2p(TEXT)
    tokens = TextCleaner()(ps)
    tokens.insert(0, 0)
    tokens_t = torch.LongTensor(tokens).unsqueeze(0)
    T = tokens_t.shape[-1]
    input_lengths = torch.LongTensor([T])
    text_mask = (torch.arange(T).unsqueeze(0) + 1 > input_lengths.unsqueeze(1))  # all False (len==T)

    caps = {}
    with torch.no_grad():
        t_en = text_encoder(tokens_t, input_lengths, text_mask)              # [1,512,T]
        bert_dur = bert(tokens_t, attention_mask=(~text_mask).int())         # [1,T,768]
        d_en = bert_encoder(bert_dur).transpose(-1, -2)                      # [1,512,T]
        d = predictor.text_encoder(d_en, s, input_lengths, text_mask)       # [1,T,640]
        x, _ = predictor.lstm(d)
        duration = torch.sigmoid(predictor.duration_proj(x)).sum(axis=-1)   # [1,T]
        pred_dur = torch.round(duration.squeeze()).clamp(min=1).long()      # [T]
        F = int(pred_dur.sum())
        aln = torch.zeros(T, F)
        c = 0
        for i in range(T):
            aln[i, c:c + pred_dur[i]] = 1
            c += int(pred_dur[i])
        en = d.transpose(-1, -2) @ aln.unsqueeze(0)                         # [1,640,F]
        F0_pred, N_pred = predictor.F0Ntrain(en, s)                         # [1,F],[1,F]
        asr = t_en @ aln.unsqueeze(0)                                       # [1,512,F]
        asr_new = torch.zeros_like(asr)                                     # hifigan 1-frame shift
        asr_new[:, :, 0] = asr[:, :, 0]
        asr_new[:, :, 1:] = asr[:, :, :-1]
        asr = asr_new

        orig = (torch.rand, torch.randn_like)
        torch.rand = lambda *a, **k: torch.zeros(*a, dtype=torch.float32)
        torch.randn_like = lambda x, **k: torch.zeros_like(x)
        try:
            audio = decoder(asr, F0_pred, N_pred, ref)
        finally:
            torch.rand, torch.randn_like = orig

    caps.update(tokens=np.array(tokens, dtype=np.int64), ref_s=ref_s, s_prosodic=s, ref_acoustic=ref,
                t_en=t_en, bert_dur=bert_dur, d_en=d_en, d=d, duration=duration, pred_dur=pred_dur,
                en=en, F0=F0_pred, N=N_pred, asr=asr, audio=audio)

    # ---- fold weight_norm and dump ALL module weights for the Rust port ----
    for mod in (text_encoder, predictor, decoder):
        for m in mod.modules():
            if hasattr(m, "weight_g"):
                torch.nn.utils.remove_weight_norm(m)
    aw = {}
    # decoder weights are dumped UNPREFIXED (StyleTtsDecoder looks them up raw, matching the
    # isolation fixtures); no key collisions with the acoustic-module prefixes.
    for pfx, mod in (("text_encoder", text_encoder), ("bert", bert), ("bert_encoder", bert_encoder), ("predictor", predictor), ("", decoder)):
        for k, v in mod.state_dict().items():
            if "num_batches_tracked" in k or k.endswith("position_ids"):
                continue
            aw[f"{pfx}.{k}" if pfx else k] = v
    for k, v in aw.items():
        v.detach().cpu().float().numpy().astype("<f4").tofile(os.path.join(OUT, "bin", f"{k}.bin"))
    print("module weights dumped:", len(aw))
    print("  text_encoder sample:", [k for k in aw if k.startswith("text_encoder")][:8])
    print("  predictor sample:", [k for k in aw if k.startswith("predictor")][:10])

    shapes = {}
    for k, v in caps.items():
        arr = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        (arr.astype("<i8") if arr.dtype == np.int64 else arr.astype("<f4")).tofile(os.path.join(OUT, "bin", f"{k}.bin"))
        shapes[k] = list(arr.shape)

    import soundfile as sf
    sf.write(os.path.join(OUT, "cloned.wav"), audio.squeeze().cpu().numpy(), 24000)

    meta = {"text": TEXT, "phonemes": ps, "tokens": tokens, "T": T, "F": F,
            "pred_dur": pred_dur.tolist(), "n_samples": int(audio.numel()),
            "ref_wav": REF_WAV, "shapes": shapes}
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"phonemes: {ps!r}")
    print("stages:", ", ".join(f"{k}{shapes[k]}" for k in caps))
    print(f"wrote synth fixtures + cloned.wav ({audio.numel()} samp) → {OUT}")


if __name__ == "__main__":
    sys.exit(main())
