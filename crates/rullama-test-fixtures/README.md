# rullama-test-fixtures

**Internal test fixtures for [rullama](https://github.com/Brainwires/rullama-framework).**

Shared mock providers, recording wrappers, tool registries, and sandbox helpers
used across the workspace's crate test suites. Consolidates mock implementations
that were previously duplicated inline across multiple `rullama-*` test modules.

> Not published to crates.io. Intended for use in `#[cfg(test)]` blocks and
> `tests/` directories within the rullama workspace (consumed as a
> dev-dependency).

## Contents

- `provider` — mock `Provider` implementations and recording wrappers for
  asserting on the exact requests a crate makes to an LLM backend.

## Usage

```toml
[dev-dependencies]
rullama-test-fixtures = { workspace = true }
```

## License

MIT OR Apache-2.0.
