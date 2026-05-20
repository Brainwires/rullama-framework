# brainwires-wasm (DEPRECATED)

This crate was **moved out of the publishable framework set** in v0.10.
The wasm bindings now live as a non-published extra at
`extras/brainwires-wasm/` inside the framework workspace (`publish =
false`), and consumer code generally goes through the `brainwires`
facade with the `wasm` feature.

There is **no replacement on crates.io.** If you previously depended on
this crate, switch to either:

| Use case | Replacement |
|---|---|
| Library use of the framework with wasm-compatible features | [`brainwires`](https://crates.io/crates/brainwires) with `features = ["wasm"]` (and other features as needed) |
| Building the browser-bindings crate yourself | `extras/brainwires-wasm/` inside the framework repo (not on crates.io) |

## Migration

### Cargo.toml

```toml
# Before
brainwires-wasm = "0.8"

# After — for most consumers
brainwires = { version = "0.11", default-features = false, features = ["wasm"] }
```

If you were using the wasm-bindgen surface specifically, build the
`extras/brainwires-wasm` crate from source — it's intentionally
unpublished. See [CHANGELOG.md](https://github.com/Brainwires/brainwires-framework/blob/main/CHANGELOG.md).
