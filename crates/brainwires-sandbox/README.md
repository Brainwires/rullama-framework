# brainwires-sandbox

Container-based sandboxing for tool execution in the Brainwires Agent
Framework. Provides a `Sandbox` trait with Docker / Podman implementations
(via [bollard](https://crates.io/crates/bollard)) that isolate tool
invocations from the host: resource limits, read-only root filesystems,
egress-allowlist networking, and whitelisted bind mounts.

```
┌──────────────────────┐     spawn()        ┌─────────────────────┐
│  ChatAgent /         │ ──────────────▶    │  SandboxPolicy      │
│  brainwires-tool-*   │                    │  - resource caps    │
└──────────────────────┘                    │  - mount whitelist  │
                                            │  - NetworkPolicy    │
                                            └──────────┬──────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────────┐
                                            │  DockerSandbox      │
                                            │  (bollard client)   │
                                            └──────────┬──────────┘
                                                       │
                                        ┌──────────────┼──────────────┐
                                        ▼              ▼              ▼
                                ┌──────────────┐┌──────────────┐┌──────────────┐
                                │  tool        ││  sidecar     ││  internal    │
                                │  container   ││  proxy       ││  docker net  │
                                │  (read-only) ││  (allowlist) ││  (no egress) │
                                └──────────────┘└──────────────┘└──────────────┘
```

## Features

| Flag           | Default | Enables                                                                  |
|----------------|---------|--------------------------------------------------------------------------|
| `docker`       | on      | `DockerSandbox` (bollard-backed; works with Docker and Podman sockets). |
| `unsafe-host`  | off     | `HostSandbox` pass-through. **Development only.** No isolation.         |

## Quick start

```rust
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;
use brainwires_sandbox::{
    DockerSandbox, ExecSpec, NetworkPolicy, Sandbox, SandboxPolicy,
};

# async fn demo() -> brainwires_sandbox::Result<()> {
let policy = SandboxPolicy {
    image: "ghcr.io/brainwires/brainclaw-sandbox:latest".into(),
    network: NetworkPolicy::Limited(vec!["pypi.org".into()]),
    memory_limit_mb: Some(512),
    cpu_limit: Some(1.0),
    pid_limit: Some(128),
    ..SandboxPolicy::default()
};

let sandbox = DockerSandbox::connect(policy)?;
let spec = ExecSpec {
    cmd: vec!["python".into(), "-c".into(), "print(2 + 2)".into()],
    env: BTreeMap::new(),
    workdir: PathBuf::from("/"),
    stdin: None,
    mounts: vec![],
    timeout: Duration::from_secs(10),
};
let handle = sandbox.spawn(spec).await?;
let out = sandbox.wait(handle).await?;
assert_eq!(out.exit_code, 0);
# Ok(()) }
```

## API

### `Sandbox` trait

```rust
#[async_trait]
pub trait Sandbox: Send + Sync {
    async fn spawn(&self, spec: ExecSpec) -> Result<ExecHandle>;
    async fn wait(&self, handle: ExecHandle) -> Result<ExecOutput>;
    async fn shutdown(&self) -> Result<()>;
    fn runtime(&self) -> SandboxRuntime;
}
```

- **`spawn`** — create the container, attach stdio, validate every mount
  against the policy (path components are checked for `..` traversal **and**
  each source is `canonicalize`d to close the symlink-race window), start
  the workload. Returns an opaque `ExecHandle`.
- **`wait`** — stream stdout/stderr, await the container exit. Enforces
  `ExecSpec.timeout` via `tokio::time::timeout`; on elapse, `kill_container`
  is called explicitly and `SandboxError::Timeout` is returned with
  `tracing::warn` carrying `container_id`, `elapsed_ms`, and `timeout_ms`.
- **`shutdown`** — force-remove every container still tracked by this
  sandbox instance; tears down per-spawn networks and proxy sidecars.

### `SandboxPolicy`

| Field                   | Purpose                                                         |
|-------------------------|-----------------------------------------------------------------|
| `runtime`               | `Docker`, `Podman`, or `Host` (unsafe-host feature only).       |
| `image`                 | Container image for workloads.                                  |
| `network`               | `None`, `Limited(Vec<String>)`, or `Full`.                      |
| `cpu_limit`             | Cores (e.g. `2.0`). `None` disables.                            |
| `memory_limit_mb`       | Cap in MiB. `None` disables.                                    |
| `pid_limit`             | Max processes inside the sandbox.                               |
| `read_only_rootfs`      | Default `true`.                                                 |
| `workspace_mount`       | Implicitly allowed bind source.                                 |
| `allowed_mount_sources` | Explicit host-path whitelist for bind mounts.                   |
| `proxy_image`           | Image for the `Limited` sidecar proxy.                          |
| `proxy_listen_port`     | TCP port the proxy listens on inside its container.             |
| `proxy_container_name`  | Reuse a named long-lived proxy; default spawns an ephemeral one.|

### `NetworkPolicy`

- **`None`** — `--network=none`. No egress, no DNS, no loopback to host.
- **`Full`** — default bridge. No egress controls. Trusted images only.
- **`Limited(hosts)`** — sandbox lives on an `internal: true` docker
  network with no default route. A `brainwires-sandbox-proxy` sidecar is
  attached to both that network and the bridge; the sandbox receives
  `HTTP_PROXY` / `HTTPS_PROXY` env vars. Only hosts in `hosts` are
  forwarded; raw (non-HTTP) TCP is blocked at the network level by design.

### Errors

```rust
pub enum SandboxError {
    Io(std::io::Error),
    Docker(String),          // feature = "docker"
    Timeout,                  // ExecSpec.timeout exceeded
    PolicyViolation(String),  // whitelist reject, symlink mismatch, etc.
    ExitFailure { code: i32, stderr: String },
    NotAvailable(String),
}
```

## Usage notes

**Mount validation** is two-stage: `SandboxPolicy::validate_mount` rejects
paths that aren't lexically inside the whitelist or contain `..`
components; `DockerSandbox::spawn` then `canonicalize`s each source and
re-validates the resolved path. A symlink swapped between those steps is
rejected with `SandboxError::PolicyViolation`.

**Podman** is selected by setting `policy.runtime = SandboxRuntime::Podman`.
The socket path is taken from `PODMAN_SOCKET` (default
`unix:///run/podman/podman.sock`).

**`HostSandbox`** (feature `unsafe-host`) runs the requested command
directly on the host with no isolation. It exists for local development
only and must never be enabled in production — `DockerSandbox::connect`
refuses `SandboxRuntime::Host` explicitly.

## Consumed by

`brainwires-tool-builtins` uses the `Sandbox` trait to execute bash / python tool
calls under isolation; `brainwires-agent` composes per-agent sandbox
policies.
