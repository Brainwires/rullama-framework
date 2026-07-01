# home-automation

Standalone workspace for home automation protocols (Matter, Zigbee, Z-Wave, Thread) plus the Matter commissioning tool. **Outside the framework workspace; not published.**

## Why this is here

In v0.11.0 the framework deliberately shrunk its surface — the home-automation stack carried unfinished pieces (Matter event subscription, BLE transport, NetworkCommissioning writes, Zigbee ZCL encoding gaps) that didn't belong in a stable published API. Rather than ship them as `Err("not yet implemented")` returns from `crates/rullama-hardware`, the entire `homeauto` module and its companion CLI `matter-tool` moved here.

The code is intact and continues to work. It just isn't part of the framework release cadence anymore and won't appear on crates.io until the gaps are closed.

## Layout

- `rullama-homeauto/` — the protocol stack lifted from `crates/rullama-hardware/src/homeauto/`. Features: `zigbee`, `zwave`, `thread`, `matter`, `matter-ble`, `all`.
- `matter-tool/` — the companion Matter commissioner / control CLI.

## Building

```bash
cd future/home-automation
cargo check --all-features
cargo build -p matter-tool --features ble
cargo test -p rullama-homeauto --features matter --test matter_integration
```

This is a sibling workspace — the framework's `cargo check --workspace` does **not** include it.

## Future

Either complete the open items (per-spec Matter event subscription, BLE transport, NetworkCommissioning write handling, full ZCL value-type encoding) and re-fold into the framework, or extract this directory into its own git repo. Until then it stays parked.
