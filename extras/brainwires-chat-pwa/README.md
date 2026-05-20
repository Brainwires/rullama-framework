# brainwires-chat-pwa

**Hosted at:** [chat.brainwires.dev](https://chat.brainwires.dev/)

## Overview

Installable PWA UI over the brainwires framework. Supports cloud providers
plus Hugging-Face-downloaded local models via [candle](https://github.com/huggingface/candle).
Streaming survives screen-off / tab-backgrounded states by routing through a
service worker. Mobile-first.

This extra is intentionally not published to crates.io — it is a deliverable
artifact (a static PWA bundle), not a library.

## Build

```sh
./web/build.sh
```

The script invokes `wasm-pack` against `wasm/`, then bundles JS via esbuild
and patches `sw.js` with SRI hashes. Output lands under `web/`:

- `web/pkg/brainwires_chat_pwa_bg.wasm` — wasm-pack output
- `web/pkg/brainwires_chat_pwa.js` — wasm-pack JS shim
- `web/app.js` (+ sourcemap) — bundled boot module
- `web/sw.js` — service worker with SRI table substituted
- `web/build-info.js` — build timestamp + git SHA

## Run

One launcher with daemon-mode subcommands. Every start always shuts
down any existing instance first, so switching between prod and dev
is seamless and idempotent.

```sh
./web/start.sh                # prod (default) — containers detached
./web/start.sh prod
./web/start.sh dev            # live-edit — all loops detached
./web/start.sh status         # show running state + container ps
./web/start.sh logs           # tail combined logs
./web/start.sh logs esbuild   # tail one channel — esbuild | cargo | compose | container
./web/start.sh stop           # tear everything down
./web/start.sh --help
```

State (PIDs + log files) lives in `web/.run/` (gitignored). Every
start command returns control to the shell as soon as the loops are
launched — no foreground compose process to Ctrl-C.

### Production

`./web/start.sh prod` runs `docker compose up -d --build`. The image
bakes the bundled output (`app.js`, `sw.js`, `pkg/`, `index.html`,
`manifest.json`, …) and nginx serves them with the WASM-aware CSP,
Cross-Origin Isolation headers, long-cache for `*.wasm`, and a
no-cache rule on `/sw.js`. Set `HOST_PORT` in `.env` to override the
default `8080`.

### Hot-swap deploy (minimal-downtime updates)

`./deploy.sh` rebuilds the image **while the old container keeps
serving**, then swaps. Downtime is just the nginx restart (~2–3s)
instead of the full image-build duration. Auto-rolls-back to the
previous image if the new container fails its health check
(`/manifest.json` 200).

```sh
./deploy.sh             # normal deploy
./deploy.sh --rollback  # restore the previous image
```

Use `start.sh prod` for first-time bring-up and `deploy.sh` for
subsequent updates. Don't run `deploy.sh` while dev mode is active —
it'll refuse, since the swap would clobber the dev container.

### Dev (live editing)

`./web/start.sh dev` runs three loops, all detached:

1. **esbuild** `--watch` — `web/src/*.js` → `web/app.js` + `web/sw.js`.
2. **cargo-watch + wasm-pack** — `wasm/` → `web/pkg/`.
3. **`docker compose up -d --build`** with the
   [`docker-compose.dev.yml`](./docker-compose.dev.yml) overlay — the
   overlay bind-mounts `./web` → `/usr/share/nginx/html` (read-only),
   so the freshly bundled output is served by nginx as soon as
   esbuild / wasm-pack write the file. No image rebuild for source
   changes; rebuild only when the image itself changes (Dockerfile,
   nginx.conf, etc. — bring it down and `start.sh dev` again).

With `DEV_MODE=true` (which the overlay forces and `start.sh dev`
exports), `boot.js` unregisters any existing service worker and
clears `bw-chat-cache-v1`, so HTML/CSS/JS edits hit the browser on
next reload without an image rebuild. `bw-models-v1` (downloaded
local models) is preserved.

Tail logs to follow what's happening:

```sh
./web/start.sh logs           # esbuild + cargo + container, combined
./web/start.sh logs esbuild   # just esbuild's incremental rebuilds
./web/start.sh logs cargo     # just cargo-watch / wasm-pack
./web/start.sh logs container # just nginx (docker compose logs -f)
```

Stop with `./web/start.sh stop` — kills the host watchers and
`docker compose down`s with the right `-f` chain for the recorded
mode.

### Manual mode (no `start.sh`)

Prefer to drive it yourself? Compose with the dev overlay:

```sh
DEV_MODE=true docker compose \
    -f docker-compose.yml \
    -f docker-compose.dev.yml \
    up -d --build
```

Then run `npm run watch` and a `wasm-pack` (or `cargo watch`) loop in
separate shells to regenerate the bundled outputs. `docker compose
-f docker-compose.yml -f docker-compose.dev.yml down` to stop.

### Esbuild-only dev (no Docker)

If you don't need the nginx headers / Docker pipeline, just bundle
and serve from esbuild:

```sh
cd web
npm install
npm run serve     # http://127.0.0.1:3000
```

Use `npm run watch` for incremental rebuilds without the server.

## Layout

```
extras/brainwires-chat-pwa/
├── README.md
├── wasm/                      # Rust → wasm32 crate (cdylib + rlib)
│   ├── Cargo.toml
│   └── src/lib.rs
└── web/                       # Static PWA assets + build glue
    ├── .gitignore
    ├── build.mjs              # esbuild + SRI patcher
    ├── build.sh               # one-shot pipeline (wasm-pack → bundle)
    ├── icons/                 # icon-192.png, icon-512.png
    ├── index.html
    ├── manifest.json
    ├── package.json           # devDependency: esbuild
    ├── styles.css
    ├── sw.source.js           # checked-in SW template
    ├── crypto-store.js        # passphrase-derived key + AES-GCM helpers
    ├── src/                   # boot, views, streaming, providers, voice, …
    │   ├── boot.js            # entry, bootstraps the WASM module
    │   ├── providers/         # anthropic / openai / google / ollama adapters
    │   └── …                  # db, model-store, ui-*, streaming, i18n, utils
    └── tests/
        ├── unit.test.mjs      # node --test, runs in CI
        └── e2e/e2e.test.mjs   # scaffold (currently all test.skip)
```

`web/pkg/`, `web/app.js`, `web/sw.js`, and `web/build-info.js` are all
build artifacts — they are regenerated on every `./build.sh` and stay
ignored from git.

## Tests

```sh
cd web
node --test tests/unit.test.mjs        # streaming, crypto-store, providers, db, utils
node --test tests/e2e/e2e.test.mjs     # scaffold; scenarios are skipped pending a
                                       # browser harness (Thalora / Playwright).
```

## Docker

A multi-stage Dockerfile bundles the wasm + JS pipeline and serves the
result behind nginx.

```sh
# From the workspace root:
docker build -f extras/brainwires-chat-pwa/Dockerfile -t brainwires-chat-pwa .
docker run --rm -p 8080:80 brainwires-chat-pwa
# → http://localhost:8080
```

Or via compose (run from `extras/brainwires-chat-pwa/`):

```sh
docker compose up --build
# → http://localhost:8080  (compose default)

# Or with the example overrides:
cp .env.example .env
docker compose up --build
# → http://localhost:8888
```

`.env` is git-ignored. Compose loads it automatically; anything not set
falls back to the defaults baked into `docker-compose.yml`. Useful keys:

| Var             | Compose default | `.env.example` value | Effect                                    |
|-----------------|-----------------|----------------------|-------------------------------------------|
| `HOST_PORT`     | `8080`          | `8888`               | Host-side port mapped to container `:80`  |
| `DEV_MODE`      | `false`         | `false`              | Enables debug surfaces in the PWA         |
| `BUILD_VERSION` | `0.1.0`         | (commented)          | Stamped into `build-info.js`              |
| `BUILD_COMMIT`  | (auto)          | (commented)          | Stamped into `build-info.js`              |
| `BUILD_DATE`    | (auto)          | (commented)          | Stamped into `build-info.js`              |

The image is ~30 MB at runtime: nginx:alpine plus the static bundle. The
builder stage uses `rust:1-bookworm` + Node 20 + `wasm-pack`; first-time
builds compile the workspace crates the wasm crate depends on, so expect
a few minutes. Subsequent builds reuse Docker layers.

`entrypoint.sh` rewrites `build-info.js` at container start so build
metadata and `DEV_MODE` can be flipped via env vars without rebuilding:

| Env var                       | Effect                              |
|-------------------------------|-------------------------------------|
| `BRAINWIRES_DEV_MODE`         | Sets `DEV_MODE` exported by build-info.js |
| `BRAINWIRES_BUILD_VERSION`    | Overrides `BUILD_VERSION`           |
| `BRAINWIRES_BUILD_COMMIT`     | Overrides `BUILD_GIT`               |
| `BRAINWIRES_BUILD_DATE`       | Overrides `BUILD_TIME`              |

`nginx.conf` ships with a CSP that allows `wasm-unsafe-eval` (with a
fallback for Safari ≤ 16.0), Cross-Origin-Isolation headers for
`SharedArrayBuffer`, long-cache for `*.wasm`, and a no-cache rule for
`/sw.js`. There is no backend in this image — no relay, no TURN, no
proxy. The PWA talks to LLM providers (Anthropic, OpenAI, Gemini, Ollama)
or runs Candle locally in-browser.

## Local model sources

Two parallel download paths land in OPFS (`model-downloads/`):

- **HuggingFace safetensors** (default) — `KNOWN_MODELS` registry in
  `web/src/model-store.js`. Currently `gemma-4-e2b-it` (~10 GB BF16).
  Loader chunks the safetensors file via `init_local_multimodal_chunked`
  on the wasm side so peak linear memory stays bounded.
- **Ollama-format GGUF** — `KNOWN_OLLAMA_MODELS` registry in
  `web/src/model-store.js`. Currently `gemma4:e2b` (~1.6 GB Q4_K_M,
  same model 6× smaller download). Pulled directly from
  `registry.ollama.ai` via the OCI Distribution Spec client in
  `web/src/ollama-fetch.js`. Tokenizer companion fetched from the HF
  repo when the manifest doesn't include one.

The wasm loader path (`init_local_multimodal_gguf` in
`wasm/src/gemma_pipeline.rs`) currently dequantizes Q4_K_M to BF16 at
load time, so VRAM footprint matches the safetensors path —
**inference tok/s is the same on both sources.** The smaller download
is the only user-visible win until a `quantized_gemma4` model that
consumes `QMatMul` directly is wired up. The candle WGPU backend
(via PR #3379) ships the `q4_k.pwgsl` quantized matmul kernel
already, so that work is small once the model port lands.

The `gemma4_diag` example at
`crates/brainwires-provider/examples/gemma4_diag.rs` exercises the
GGUF loader end-to-end on native:

```sh
cargo run --release -p brainwires-provider \
    --features native,local-llm-vision,candle-wgpu \
    --example gemma4_diag -- \
    --gguf-path ~/Downloads/gemma4-e2b-q4_k_m.gguf \
    --tokenizer-file ~/.cache/huggingface/hub/.../tokenizer.json \
    --device cpu --prompt "Hi" --max-new-tokens 1
```

## Constraints

No model weights are bundled. Every model is fetched from huggingface.co
or registry.ollama.ai at runtime. The crate ships only the runtime
shell — no `*.gguf`, `*.safetensors`, or `*.bin` ever ride along in
the artifact.
