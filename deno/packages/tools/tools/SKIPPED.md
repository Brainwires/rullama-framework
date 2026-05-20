# Rust tool modules intentionally not ported to Deno

These modules are part of `brainwires-tools` (Rust) but are not available in `@brainwires/tools` (Deno). Each skip has a specific technical reason below. Callers that need these capabilities should run the Rust framework and communicate with Deno agents over `@brainwires/network` / `@brainwires/a2a`.

## `crates/brainwires-tools/src/email/`

IMAP, SMTP, and Gmail-push integrations. IMAP/SMTP clients in Deno require large npm shims (`nodemailer`, `imapflow`) with their own dependency trees; Gmail push needs Google Cloud Pub/Sub. Deferred to a later phase — the Rust implementations talk directly to RFC-compliant servers and are production-ready.

## `crates/brainwires-tools/src/system/services/` (systemd, docker, process)

OS-level service management (systemd units, docker daemon control, raw process supervision). These assume a Unix host with root-capable access to a service manager. Not a good fit for a Deno runtime that's typically sandboxed; the Rust version is the correct home for this.

## `crates/brainwires-tools/src/system/reactor/`

Filesystem reactor with rule-based matching. Deno has `Deno.watchFs` but `rules.rs` builds ripgrep-style glob matchers that are heavily tied to Rust crates. Deferred — can be re-implemented in Deno with a pure-TS glob engine if needed.

## `crates/brainwires-tools/src/system/config.rs`

Thin config loader shared by the other `system/*` modules. Skipped alongside them; no standalone value without the reactor / services backends.

## `crates/brainwires-tools/src/browser.rs`

Headless-browser automation. The Deno port will route through the Thalora bridge (pure-Rust headless browser — see `reference_thalora.md`). Deferred until the Thalora ↔ Deno transport layer is defined.

## `crates/brainwires-tools/src/interpreters/`

Embedded scripting interpreters (Rhai, Boa, RustPython). These are binary-backed engines loaded as native libraries — no equivalent in Deno. The recommended Deno alternative is calling out to an Ollama local model via `OllamaChatProvider` for code-synthesis use cases.

## `crates/brainwires-tools/src/orchestrator/`

Rhai-based orchestrator that drives scripted workflows. Depends on `interpreters/` and is Rust-only by construction. Deno agents should express workflows in TypeScript directly rather than a scripting layer.

## `crates/brainwires-tools/src/sandbox_executor.rs`

Sandboxed tool execution that delegates to `brainwires-sandbox` (Docker/Podman). The sandbox crate itself is Rust-only (bollard client + container lifecycle); the executor is a thin adapter and isn't useful without it.

## `crates/brainwires-tools/src/code_exec.rs`

Multi-language code execution that dispatches to `interpreters/`. Skipped transitively.
