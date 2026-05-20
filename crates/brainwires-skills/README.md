# brainwires-skills

[![Crates.io](https://img.shields.io/crates/v/brainwires-skills.svg)](https://crates.io/crates/brainwires-skills)
[![Documentation](https://docs.rs/brainwires-skills/badge.svg)](https://docs.rs/brainwires-skills)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue)](https://github.com/Brainwires/brainwires-framework)

The SKILL.md skills system for the Brainwires Agent Framework —
manifest parsing, registry, smart routing, sandboxed execution.

## Overview

Skills are reusable units of capability defined by a `SKILL.md`
manifest. The skills system parses manifests, routes a user query
through a registry to find the best match, and executes the chosen
skill against the framework's `ToolExecutor`.

Originally a separate crate (v0.8.x), then folded into
`brainwires-agent`, then re-extracted in 0.11 (Phase 11c) once the
framework's "agent" boundary was tightened to coordination only.

## Modules

- `manifest` — `SkillManifest` schema (the SKILL.md frontmatter shape)
- `parser` — SKILL.md → `SkillManifest`
- `metadata` — `Skill`, `SkillResult`, `SkillExecutionMode`,
  `SkillSource`, `MatchSource`
- `registry` — `SkillRegistry` lookup
- `router` — `SkillRouter` for query-to-skill matching with
  `SkillMatch`
- `executor` — `SkillExecutor` runs a chosen skill
- `tool_adapter` — wraps a skill as a `Tool` for the framework's
  `ToolExecutor`
- `package` — `SkillPackage` bundling
- `verification` — manifest verification (sha256 hashing always-on,
  ed25519 signing behind the `signing` feature)
- `registry_client` — remote skills-registry HTTP client (behind the
  `registry` feature)

## Features

| Flag       | Default | Enables                                                            |
|------------|---------|--------------------------------------------------------------------|
| `registry` | off     | `SkillRegistryClient` — HTTP client for fetching skills from a remote registry |
| `signing`  | off     | ed25519 manifest signing + verification                            |

## Migration from `brainwires-agent::skills`

```toml
# Before
brainwires-agent = { features = ["skills-registry"] }

# After
brainwires-skills = { features = ["registry"] }
```

```rust
// Before
use brainwires_agent::skills::{SkillRegistry, SkillRouter};

// After
use brainwires_skills::{SkillRegistry, SkillRouter};
```

## License

MIT OR Apache-2.0
