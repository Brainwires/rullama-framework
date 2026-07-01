# rullama-fuzz

Fuzz targets attacking the most security-relevant parse/decode entry
points in the framework. Standalone workspace — not a member of the
root workspace — because `cargo-fuzz` requires nightly Rust and the
main workspace MSRV is 1.91 stable.

## Targets

| Target                  | Attacks                                         |
|-------------------------|-------------------------------------------------|
| `mcp_jsonrpc_parser`    | `rullama_mcp_client::JsonRpcRequest` decode  |
| `a2a_envelope_decoder`  | `rullama_a2a::agent_card::AgentCard` decode  |
| `skill_manifest`        | `rullama_skills::SkillManifest` decode       |
| `model_loader_header`   | stub — replace with in-tree loader once landed  |

## Running

```bash
# One-time setup (cargo-fuzz CLI)
cargo install cargo-fuzz

# Short run (CI cadence)
cd fuzz
cargo +nightly fuzz run mcp_jsonrpc_parser -- -max_total_time=60

# Long run (nightly cadence)
cargo +nightly fuzz run mcp_jsonrpc_parser -- -max_total_time=1800
```

Crash inputs land in `fuzz/artifacts/<target>/`. Once a target stops
finding new edges quickly, archive the corpus growth back into git so
subsequent runs start from a richer base.

## Adding a target

1. Drop a new `fuzz_targets/<name>.rs` using the `fuzz_target!` macro.
2. Add a matching `[[bin]]` entry in `fuzz/Cargo.toml`.
3. Seed `corpus/<name>/` with one or two minimal-but-valid inputs so
   the engine has a starting point.
