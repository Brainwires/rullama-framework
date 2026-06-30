# rullama-test-harness

**Cross-crate test harness for [rullama](https://github.com/Brainwires/rullama-framework).**

A one-way test-case producer: each tier returns `Vec<Arc<dyn EvaluationCase>>`
that can be fed into `rullama_eval::EvaluationSuite` directly, or into
`rullama_autonomy::AutonomousFeedbackLoop` (which consumes the harness output
without the harness depending on the autonomy crate).

> Not published to crates.io. Orchestrated via `cargo xtask test-harness`.

## Three tiers

- **Tier A — feature determinism.** Every `FEATURES.md` heading has at least one
  deterministic case. The manifest at `tests/feature_inventory.toml` lists the
  Rust function paths registered via `registry`.
- **Tier B — security adversarial.** Per-invariant adversarial cases registered
  via `inventory::submit!` next to each attack.
- **Tier C — golden-path assemblies.** Manually-listed integration scenarios in
  `assemblies`.

## Binaries

- `run-harness` — execute the tiers.
- `print-manifest` — dump the registered case inventory.

```sh
cargo xtask test-harness
```

## License

MIT OR Apache-2.0.
