# brainwires-skill-registry

HTTP registry server for publishing, searching, and downloading skill packages.

## Quick Start

```bash
cargo run -p brainwires-skill-registry -- serve
```

Or with options:

```bash
cargo run -p brainwires-skill-registry -- serve --listen 0.0.0.0:3000 --db skills.db
```

## CLI

| Flag | Description |
|------|-------------|
| `--listen` | Listen address (default: `0.0.0.0:3000`) |
| `--db` | SQLite database path (default: `skills.db`) |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/skills` | Publish a new skill package |
| `GET` | `/api/skills/search?q=...&tags=...&limit=...` | Search skills by query and tags |
| `GET` | `/api/skills/:name` | Get latest manifest for a skill |
| `GET` | `/api/skills/:name/versions` | List all versions of a skill |
| `GET` | `/api/skills/:name/:version` | Get manifest for a specific version |
| `GET` | `/api/skills/:name/:version/download` | Download full skill package |

## Storage

Uses SQLite with FTS5 full-text search for efficient skill discovery. The database schema is auto-created on first run.

## Skill Package Format

Skills are distributed as `SkillPackage` JSON containing:
- **manifest** — name, semver version, author, license, tags, dependencies
- **skill_content** — the SKILL.md file content
- **checksum** — SHA-256 integrity hash
- **signature** — optional ed25519 signature (when `signing` feature enabled)

## Publishing from CLI

Use the `RegistryClient` from `brainwires-agent::skills` (formerly the standalone `brainwires-skills` crate, absorbed in the 0.10 consolidation):

```rust
use brainwires_agent::skills::{RegistryClient, SkillPackage, SkillManifest};

let client = RegistryClient::new("http://localhost:3000", None);
let package = SkillPackage::from_skill_file("my-skill/SKILL.md", manifest)?;
client.publish(&package).await?;
```
