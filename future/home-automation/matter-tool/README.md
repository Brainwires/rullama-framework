# matter-tool

A first-party Matter 1.3 CLI for the Brainwires framework — the `chip-tool` equivalent built
entirely on the Brainwires pure-Rust Matter stack. No `connectedhomeip` dependency; compiles in
seconds.

## Installation

```bash
cargo install --path extras/matter-tool
# or with BLE commissioning support:
cargo install --path extras/matter-tool --features ble
```

## Usage

```
matter-tool [OPTIONS] <COMMAND>
```

### Global options

| Flag | Description |
|------|-------------|
| `--fabric-dir <DIR>` | Fabric storage directory (default: `~/.local/share/matter-tool/`) |
| `-v`, `--verbose` | Enable debug-level tracing |
| `--json` | Machine-readable JSON output |

### Commissioning (pair)

```bash
# Commission via QR code
matter-tool pair qr 1 "MT:Y.K9042C00KA0648G00"

# Commission via 11-digit manual pairing code
matter-tool pair code 1 34970112332

# Remove from fabric (local only, no network interaction)
matter-tool pair unpair 1
```

### OnOff cluster

```bash
matter-tool onoff on   1 1
matter-tool onoff off  1 1
matter-tool onoff toggle 1 1
matter-tool onoff read 1 1
```

### Level control

```bash
matter-tool level set 1 1 128           # level 0–254
matter-tool level set 1 1 200 --transition 10   # 1.0 s transition
matter-tool level read 1 1
```

### Thermostat

```bash
matter-tool thermostat setpoint 1 1 21.5
matter-tool thermostat read 1 1
```

### Door lock

```bash
matter-tool doorlock lock   1 1
matter-tool doorlock unlock 1 1
matter-tool doorlock read   1 1
```

### Raw invoke / read

```bash
# Send raw cluster command (payload-hex is optional TLV bytes)
matter-tool invoke 1 1 0x0006 0x01

# Read raw attribute
matter-tool read 1 1 0x0006 0x0000
```

### Discovery

```bash
# Browse for commissionable and operational Matter devices (5 s)
matter-tool discover

# Extend discovery window
matter-tool discover --timeout 15
```

### Device server (serve)

```bash
# Run as a Matter device; commission us from any Matter controller
matter-tool serve --device-name "Test Light" --vendor-id 0xFFF1 --product-id 0x8001
```

Prints the QR code and manual pairing code on startup. Press Ctrl-C to stop.

### Fabric management

```bash
matter-tool devices          # list commissioned devices
matter-tool fabric info      # fabric directory + node count
matter-tool fabric reset     # wipe all fabric storage (asks "yes" to confirm)
```

## JSON output

Append `--json` to any command for machine-readable output:

```bash
matter-tool --json devices
matter-tool --json read 1 1 0x0006 0x0000
matter-tool --json discover
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| *(none)* | Yes | QR/manual commissioning, all cluster commands |
| `ble` | No | BLE commissioning (`pair ble`) via btleplug |
