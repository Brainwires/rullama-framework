# Kokoro-82M reference (1:1 porting spec)

Source of truth for the TTS port. Mirrors the upstream `hexgrad/kokoro` Python
package (StyleTTS2 acoustic model + ISTFTNet vocoder) the same way the Gemma ops
mirror Ollama's `model/models/gemma4/`. When porting an op, diff against the
upstream Python, not a llama.cpp/other port.

- Upstream code (reference copy): `hexgrad/kokoro` → `kokoro/{model,modules,istftnet,custom_stft,pipeline}.py`
- Weights (local dev cache): `~/.cache/kokoro/kokoro-v1_0.pth` (327 MB), `~/.cache/kokoro/voices/af_heart.pt`
- Config: `~/.cache/kokoro/config.json`
- License: Apache-2.0 (model + code). 24 kHz mono output.

## Config (config.json)

```
n_token = 178          # phoneme vocab size (IPA + punctuation; see vocab map below)
hidden_dim = 512       # d_hid throughout predictor/text_encoder
style_dim = 128        # voice/style vector half-size (full ref_s = 256 = 2×128)
dim_in = 64            # decoder input feature dim (== asr_res out channels)
n_mels = 80            # decoder dim_out (NOT used as a literal mel count in the iSTFT head)
n_layer = 3            # depth of text_encoder CNN and DurationEncoder LSTM/AdaLN stack
max_dur = 50           # duration_proj output width (per-token duration logits)
text_encoder_kernel_size = 5
max_conv_dim = 512
dropout = 0.2          # inference: all dropout is identity

plbert: hidden_size=768, num_attention_heads=12, intermediate_size=2048,
        max_position_embeddings=512, num_hidden_layers=12, dropout=0.1   # ALBERT (shared weights!)

istftnet: upsample_rates=[10,6], upsample_kernel_sizes=[20,12],
          upsample_initial_channel=512,
          resblock_kernel_sizes=[3,7,11], resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]],
          gen_istft_n_fft=20, gen_istft_hop_size=5
```

`context_length = max_position_embeddings = 512`. Input is wrapped `[0, *ids, 0]`
(BOS/EOS = id 0), so max ~510 phonemes/utterance.

## Top-level module tree (== `.pth` state-dict keys)

`kokoro-v1_0.pth` is `{submodule_name: state_dict}` for exactly these 5 keys:

| key            | module           | shape notes |
|----------------|------------------|-------------|
| `bert`         | `CustomAlbert`   | HF ALBERT; **all 12 layers share one weight set** (ALBERT param sharing) |
| `bert_encoder` | `nn.Linear`      | 768 → 512 |
| `predictor`    | `ProsodyPredictor` | duration + F0/N |
| `text_encoder` | `TextEncoder`    | phoneme → acoustic text embedding |
| `decoder`      | `Decoder`        | AdaIN decode stack + ISTFTNet `Generator` |

**GGUF converter:** enumerate each state_dict, keep the PyTorch param path as the
tensor name under a short prefix (e.g. `k.bert.*`, `k.bertenc.*`, `k.pred.*`,
`k.tenc.*`, `k.dec.*`), emit F16/F32 (NOT Q4_K). Voicepacks (`voices/<id>.pt`,
shape `[510,1,256]`) → tensors `k.voice.<id>`. Set `general.architecture="kokoro"`,
`kokoro.*` metadata keys parsed by a new `KokoroConfig` (mirror `Gemma4Config`).

## Forward dataflow (`KModel.forward_with_tokens`, model.py)

Input: `input_ids` `[1,T]` (T incl. BOS/EOS), `ref_s` `[1,256]`, `speed=1`.

```
text_mask        = all-false for the single full-length sequence (no padding at B=1)
bert_dur [1,T,768] = bert(input_ids, attention_mask=~text_mask)        # ALBERT
d_en    [1,512,T]  = bert_encoder(bert_dur).transpose(-1,-2)
s       [1,128]    = ref_s[:, 128:]                                    # PROSODIC half
d       [1,T,512]  = predictor.text_encoder(d_en, s, lengths, text_mask)  # DurationEncoder
x,_                = predictor.lstm(d)                                  # BiLSTM 512+128→512
duration[1,T,50]   = predictor.duration_proj(x)
dur     [1,T]      = round(sigmoid(duration).sum(-1) / speed).clamp(min=1).long()
# length regulator: expand each token t to dur[t] frames
pred_aln_trg [T, Lf] = one-hot expansion (indices = repeat_interleave(arange(T), dur))
en      [1,512,Lf]  = d.transpose(-1,-2) @ pred_aln_trg
F0,N    [1,Lf']     = predictor.F0Ntrain(en, s)                        # shared BiLSTM + AdainResBlk1d stacks
t_en    [1,512,T]   = text_encoder(input_ids, lengths, text_mask)
asr     [1,512,Lf]  = t_en @ pred_aln_trg                              # aligned acoustic features
audio   [Lsamp]     = decoder(asr, F0, N, ref_s[:, :128])             # :128 = TIMBRE half
```

**Voice split is load-bearing:** `ref_s[:, 128:]` (prosodic) conditions the
predictor; `ref_s[:, :128]` (timbre) conditions the decoder. The voicepack row is
selected by **token length** index: `ref_s = voice[len(input_ids)-1]` (see pipeline.py).

## Module internals

### TextEncoder (modules.py)
`embedding(n_token,512)` → transpose → `n_layer=3 ×` [ `weight_norm Conv1d(512,512,k=5,pad=2)` → `LayerNorm(512)` (channel-axis, with γ=`gamma`,β=`beta`) → `LeakyReLU(0.2)` ] → BiLSTM(512 → 256×2). Masked-fill 0 at padding after each step (no-op at B=1).

### ProsodyPredictor (modules.py)
- `text_encoder` = **DurationEncoder**: `n_layer ×` [ BiLSTM(512+128 → 256×2) , **AdaLayerNorm**(style 128 → 512, applies `(1+γ)·LN(x)+β` then re-concats style ] . Note style `s` is concatenated back after each AdaLN.
- `lstm`: BiLSTM(512+128 → 256×2) for durations
- `duration_proj`: Linear(512 → 50), then `sigmoid().sum(-1)` → scalar duration/token
- `shared`: BiLSTM(512+128 → 256×2) feeding F0 and N branches
- `F0` / `N`: each is 3× `AdainResBlk1d` (512→512, 512→256 **upsample=True**, 256→256), then `F0_proj`/`N_proj` = Conv1d(256→1)

### Decoder (istftnet.py)
```
F0 = F0_conv(F0_curve)      # Conv1d(1,1,k=3,stride=2,pad=1)  -> downsample ×2
N  = N_conv(N)              # Conv1d(1,1,k=3,stride=2,pad=1)
x  = cat([asr, F0, N], dim=1)                      # 512+1+1 = 514
x  = encode(x, s)                                  # AdainResBlk1d(514 -> 1024)
asr_res = asr_res(asr)                             # Conv1d(512 -> 64)
for block in decode[0..3]:                         # AdainResBlk1d(1024+2+64 -> 1024 ×3, last -> 512 upsample=True)
    if not yet upsampled: x = cat([x, asr_res, F0, N], dim=1)
    x = block(x, s)
x = generator(x, s, F0_curve)                      # ISTFTNet
```

### AdainResBlk1d (decode/predictor blocks)
`AdaIN1d` = `InstanceNorm1d(affine=True)` then style FC(128→2C) → `(1+γ)·norm(x)+β`.
Residual: `norm1→LeakyReLU(0.2)→pool(Identity or wn ConvTranspose1d stride2 for upsample)→conv1(wn 3,pad1)→norm2→LeakyReLU→conv2(wn 3,pad1)`; output `(residual + shortcut)·rsqrt(2)`. Shortcut upsamples (nearest ×2) + optional 1×1 conv when `dim_in≠dim_out`.

### AdaINResBlock1 (generator resblocks) — uses **Snake1D**, not LeakyReLU
3 parallel (conv1[dilated 1/3/5] / conv2[dilation 1]) pairs, each with AdaIN before a **Snake** activation: `x + (1/α)·sin(α·x)²` (per-channel learnable `alpha1`/`alpha2`). Residual add per pair.

### Generator / ISTFTNet (istftnet.py)
- **HnNSF source** (`SourceModuleHnNSF`+`SineGen`, harmonic_num=8, voiced_threshold=10): upsamples F0 by `prod(upsample_rates)*hop = 60·5=300`, makes harmonic sines + noise → `har_source` → STFT → `har = cat([mag, phase])` (`n_fft+2 = 22` chans). **NON-DETERMINISTIC**: random initial phase (`torch.rand`) + Gaussian noise (`torch.randn_like`). Port a **seeded/zeroed** variant for parity; expose seed.
- `ups`: 2× `weight_norm ConvTranspose1d` (rates 10,6; kernels 20,12) 512→256→128.
- per stage: `noise_convs[i]` (Conv1d 22→ch) + `noise_res[i]` (AdaINResBlock1) injected; 3 resblocks (k=3,7,11) summed/avg.
- `conv_post`: `weight_norm Conv1d(ch → n_fft+2=22, k=7, pad=3)` after `reflection_pad(1,0)` on last upsample.
- output split: `spec = exp(x[:11])`, `phase = sin(x[11:])` → `stft.inverse(spec, phase)`.

## STFT/iSTFT decision (see plan)
Port the **exact complex-equivalent** iSTFT (the `TorchSTFT`/`disable_complex=False`
path: one-sided-bin doubling + COLA window normalization), in **real arithmetic**
(no complex type; reuse the real FFT in `multimodal/audio_features.rs`). Do NOT copy
`CustomSTFT` — it's a deliberately ONNX-friendly *approximation* (its own comments
admit it skips DC/Nyquist doubling) and we have no ONNX constraint. `n_fft=20`,
`hop=5` makes exact vs. approximate free. Hann window, `periodic=True`, `center=True`.

## Porting gotchas
- **weight_norm everywhere** (`weight_g`/`weight_v`): fold to plain `weight` at convert time → no runtime weight-norm kernel.
- **ALBERT weight sharing**: 12 layers reuse ONE transformer block's weights + an `embedding_hidden_mapping_in` (128→768) projection. Store once.
- **LayerNorm (modules.LayerNorm + ALBERT)** has mean-subtraction + bias (γ/β) — distinct from Gemma RMSNorm; needs a new kernel.
- **InstanceNorm1d** (AdaIN) normalizes over the **time** axis per channel.
- **Two activation families**: LeakyReLU(0.2) in acoustic blocks, **Snake** in the vocoder resblocks.
- **Conv strides**: `F0_conv`/`N_conv` stride-2 downsample; `ConvTranspose1d` upsamples (ups + AdainResBlk1d pool).
- **Length regulator** = one-hot `[T,Lf]` from `repeat_interleave(arange(T), dur)`; `en/asr = feat @ aln`.
- **Determinism**: vocoder source noise (above) — seed it.
- All masking is a no-op at batch=1 (our case); skip the mask plumbing in v1.

## G2P (pipeline.py → misaki)
English path: `misaki.en.G2P` → IPA string → `vocab.get(ch)` per char (drop misses).
v1 plan: lexicon-first; reject OOV with a warning (espeak-ng fallback is a C dep, deferred).
Validate the Rust G2P by phoneme-string diff vs `misaki` on a fixed corpus.

## Vocab (phoneme → id), from config.json
178 entries: punctuation (`;:,.!?—…"()`…` `=16` space) + ASCII letters (mostly stress/caps
markers) + IPA blocks (`ɑɐɒæβɔ…`) + suprasegmentals (`ˈˌːʰʲ`) + tones (`↓→↗↘`). Stored in
`KokoroConfig`. (Full map in upstream `config.json["vocab"]`.)
