//! Docker / Podman sandbox backed by [bollard].
//!
//! Containers are created with resource limits, a read-only root filesystem,
//! and no inherited host environment. Only mounts permitted by the
//! [`SandboxPolicy`] whitelist are forwarded to the container runtime.
//!
//! # Network policies
//!
//! - [`NetworkPolicy::None`][]: `--network=none`.
//! - [`NetworkPolicy::Full`][]: bridge network, no egress controls.
//! - [`NetworkPolicy::Limited`][]: the sandbox container is attached to a
//!   freshly-created `internal: true` docker network with no default route.
//!   A sidecar `brainwires-sandbox-proxy` container is attached to BOTH that
//!   internal network and the default bridge; the sandbox receives
//!   `HTTP_PROXY`/`HTTPS_PROXY` env vars pointing at the proxy. The proxy
//!   enforces a per-host allowlist; raw (non-HTTP) TCP is inherently blocked
//!   because the internal network has no external route.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bollard::Docker;
use bollard::container::{
    AttachContainerOptions, AttachContainerResults, Config, CreateContainerOptions, LogOutput,
    RemoveContainerOptions, StartContainerOptions,
};
use bollard::models::{EndpointSettings, HostConfig};
use bollard::network::{ConnectNetworkOptions, CreateNetworkOptions};
use futures::StreamExt;
use tokio::sync::Mutex;
use tokio::time::timeout;

use crate::error::{Result, SandboxError};
use crate::{
    ExecHandle, ExecOutput, ExecSpec, Mount, NetworkPolicy, Sandbox, SandboxPolicy, SandboxRuntime,
};

/// Per-spawn Limited-mode sidecar state that must be torn down with the
/// sandbox container.
struct LimitedSidecar {
    network_id: String,
    /// Proxy container ID. `None` when a shared proxy (named via
    /// `SandboxPolicy::proxy_container_name`) is reused across spawns — in
    /// that case the sandbox only disconnects from the network; it does not
    /// remove the proxy.
    ephemeral_proxy: Option<String>,
    /// Shared proxy container ID (set when a named long-lived proxy was
    /// reused). Used so `wait()` can disconnect it from the per-spawn
    /// network during cleanup.
    shared_proxy: Option<String>,
}

struct Job {
    container_id: String,
    started: Instant,
    timeout: Duration,
    attach: AttachContainerResults,
    limited: Option<LimitedSidecar>,
}

/// Sandbox backed by a Docker- or Podman-compatible daemon.
pub struct DockerSandbox {
    client: Arc<Docker>,
    policy: SandboxPolicy,
    jobs: Arc<Mutex<HashMap<ExecHandle, Job>>>,
}

impl DockerSandbox {
    /// Connect to the configured daemon. For [`SandboxRuntime::Podman`] the
    /// socket path is taken from the `PODMAN_SOCKET` environment variable and
    /// falls back to `unix:///run/podman/podman.sock`.
    pub fn connect(policy: SandboxPolicy) -> Result<Self> {
        let client = match policy.runtime {
            SandboxRuntime::Docker => Docker::connect_with_socket_defaults()?,
            SandboxRuntime::Podman => {
                let socket = std::env::var("PODMAN_SOCKET")
                    .unwrap_or_else(|_| "unix:///run/podman/podman.sock".to_string());
                Docker::connect_with_socket(&socket, 120, bollard::API_DEFAULT_VERSION)?
            }
            SandboxRuntime::Host => {
                return Err(SandboxError::NotAvailable(
                    "DockerSandbox cannot run SandboxRuntime::Host; use HostSandbox instead".into(),
                ));
            }
        };

        Ok(Self {
            client: Arc::new(client),
            policy,
            jobs: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn build_host_config(&self, mounts: &[Mount], network_mode: Option<String>) -> HostConfig {
        let memory = self
            .policy
            .memory_limit_mb
            .map(|mb| (mb as i64).saturating_mul(1024 * 1024));
        let nano_cpus = self
            .policy
            .cpu_limit
            .map(|cores| (cores * 1_000_000_000f64) as i64);
        let pids_limit = self.policy.pid_limit.map(|n| n as i64);

        let binds: Vec<String> = mounts
            .iter()
            .map(|m| {
                format!(
                    "{}:{}:{}",
                    m.source.display(),
                    m.target.display(),
                    if m.read_only { "ro" } else { "rw" }
                )
            })
            .collect();

        HostConfig {
            memory,
            nano_cpus,
            pids_limit,
            network_mode,
            binds: if binds.is_empty() { None } else { Some(binds) },
            readonly_rootfs: Some(self.policy.read_only_rootfs),
            auto_remove: Some(false),
            ..Default::default()
        }
    }

    /// Create an internal (no-egress) docker network and attach the proxy to
    /// it. Returns (network_id, network_name, proxy_ip_on_network, sidecar).
    async fn setup_limited_network(
        &self,
        handle: &ExecHandle,
        hosts: &[String],
    ) -> Result<(String, String, String, LimitedSidecar)> {
        let net_name = format!("brainwires-sandbox-net-{}", handle.as_uuid());
        let create = CreateNetworkOptions::<String> {
            name: net_name.clone(),
            driver: "bridge".to_string(),
            internal: true,
            ..Default::default()
        };
        let net = self.client.create_network(create).await?;
        let network_id = net.id.unwrap_or_else(|| net_name.clone());

        let allow_env = format!("PROXY_ALLOW_HOSTS={}", hosts.join(","));
        let listen_env = format!("PROXY_LISTEN=0.0.0.0:{}", self.policy.proxy_listen_port);

        let (proxy_id, is_ephemeral) = match self
            .ensure_proxy_on_network(&network_id, &allow_env, &listen_env)
            .await
        {
            Ok(p) => p,
            Err(e) => {
                let _ = self.client.remove_network(&network_id).await;
                return Err(e);
            }
        };

        let ip = match self.proxy_ip_on_network(&proxy_id, &network_id).await {
            Ok(ip) => ip,
            Err(e) => {
                let sidecar = LimitedSidecar {
                    network_id: network_id.clone(),
                    ephemeral_proxy: if is_ephemeral {
                        Some(proxy_id.clone())
                    } else {
                        None
                    },
                    shared_proxy: if is_ephemeral {
                        None
                    } else {
                        Some(proxy_id.clone())
                    },
                };
                self.teardown_limited(sidecar).await;
                return Err(e);
            }
        };

        tracing::info!(
            allow_hosts = %hosts.join(","),
            proxy_container = %proxy_id,
            network = %network_id,
            "DockerSandbox Limited mode ready"
        );

        let sidecar = LimitedSidecar {
            network_id: network_id.clone(),
            ephemeral_proxy: if is_ephemeral {
                Some(proxy_id.clone())
            } else {
                None
            },
            shared_proxy: if is_ephemeral { None } else { Some(proxy_id) },
        };

        Ok((network_id, net_name, ip, sidecar))
    }

    /// Returns `(proxy_container_id, is_ephemeral)`. When `is_ephemeral`
    /// is false, the caller MUST NOT remove the container during cleanup
    /// (it belongs to a shared named proxy).
    async fn ensure_proxy_on_network(
        &self,
        network_id: &str,
        allow_env: &str,
        listen_env: &str,
    ) -> Result<(String, bool)> {
        let (id, is_ephemeral) = if let Some(name) = &self.policy.proxy_container_name {
            match self.client.inspect_container(name, None).await {
                Ok(info) => (info.id.unwrap_or_else(|| name.clone()), false),
                Err(bollard::errors::Error::DockerResponseServerError {
                    status_code: 404, ..
                }) => {
                    // Create the named shared proxy on first use.
                    let new_id = self
                        .create_proxy_container(Some(name.clone()), allow_env, listen_env)
                        .await?;
                    (new_id, false)
                }
                Err(e) => return Err(e.into()),
            }
        } else {
            let new_id = self
                .create_proxy_container(None, allow_env, listen_env)
                .await?;
            (new_id, true)
        };

        self.client
            .connect_network(
                network_id,
                ConnectNetworkOptions::<String> {
                    container: id.clone(),
                    endpoint_config: EndpointSettings::default(),
                },
            )
            .await?;
        Ok((id, is_ephemeral))
    }

    async fn create_proxy_container(
        &self,
        name: Option<String>,
        allow_env: &str,
        listen_env: &str,
    ) -> Result<String> {
        let cfg: Config<String> = Config {
            image: Some(self.policy.proxy_image.clone()),
            env: Some(vec![allow_env.to_string(), listen_env.to_string()]),
            host_config: Some(HostConfig {
                // Default bridge so the proxy can actually reach the
                // internet. The internal per-spawn network is attached
                // separately.
                network_mode: Some("bridge".to_string()),
                auto_remove: Some(false),
                ..Default::default()
            }),
            ..Default::default()
        };
        let opts = name.map(|n| CreateContainerOptions::<String> {
            name: n,
            platform: None,
        });
        let created = self.client.create_container(opts, cfg).await?;
        self.client
            .start_container(&created.id, None::<StartContainerOptions<String>>)
            .await?;
        Ok(created.id)
    }

    async fn proxy_ip_on_network(&self, proxy_id: &str, network_id: &str) -> Result<String> {
        // Docker assigns the IP asynchronously on network connect. Poll
        // inspect_container briefly until the Networks map contains our
        // network with a non-empty IP.
        for _ in 0..20 {
            let info = self.client.inspect_container(proxy_id, None).await?;
            if let Some(ns) = info.network_settings.as_ref()
                && let Some(map) = ns.networks.as_ref()
            {
                for (_, ep) in map
                    .iter()
                    .filter(|(_, ep)| ep.network_id.as_deref() == Some(network_id))
                {
                    if let Some(ip) = ep.ip_address.as_ref()
                        && !ip.is_empty()
                    {
                        return Ok(ip.clone());
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Err(SandboxError::NotAvailable(format!(
            "proxy container {proxy_id} never got an IP on network {network_id}"
        )))
    }

    async fn teardown_limited(&self, sidecar: LimitedSidecar) {
        if let Some(proxy_id) = sidecar.ephemeral_proxy {
            let _ = self
                .client
                .remove_container(
                    &proxy_id,
                    Some(RemoveContainerOptions {
                        force: true,
                        v: true,
                        link: false,
                    }),
                )
                .await;
        } else if let Some(shared_id) = sidecar.shared_proxy {
            // Leave the shared proxy running, but disconnect it from this
            // spawn's internal network so the network can be removed.
            let _ = self
                .client
                .disconnect_network(
                    &sidecar.network_id,
                    bollard::network::DisconnectNetworkOptions::<String> {
                        container: shared_id,
                        force: true,
                    },
                )
                .await;
        }
        let _ = self.client.remove_network(&sidecar.network_id).await;
    }
}

#[async_trait::async_trait]
impl Sandbox for DockerSandbox {
    async fn spawn(&self, spec: ExecSpec) -> Result<ExecHandle> {
        for m in &spec.mounts {
            self.policy.validate_mount(m)?;
        }

        let mut spec = spec;
        for m in spec.mounts.iter_mut() {
            let resolved = std::fs::canonicalize(&m.source).map_err(|e| {
                SandboxError::PolicyViolation(format!(
                    "mount source {} could not be canonicalized: {e}",
                    m.source.display()
                ))
            })?;
            self.policy.validate_mount(&Mount {
                source: resolved.clone(),
                target: m.target.clone(),
                read_only: m.read_only,
            })?;
            m.source = resolved;
        }

        let handle = ExecHandle::new();

        let mut env: Vec<String> = spec.env.iter().map(|(k, v)| format!("{k}={v}")).collect();

        let (network_mode, limited) = match &self.policy.network {
            NetworkPolicy::None => (Some("none".to_string()), None),
            NetworkPolicy::Full => (Some("bridge".to_string()), None),
            NetworkPolicy::Limited(hosts) => {
                let (_network_id, net_name, proxy_ip, sidecar) =
                    self.setup_limited_network(&handle, hosts).await?;
                let proxy_url = format!("http://{}:{}", proxy_ip, self.policy.proxy_listen_port);
                // Both upper- and lower-case — many CLIs only honour one.
                env.push(format!("HTTP_PROXY={proxy_url}"));
                env.push(format!("HTTPS_PROXY={proxy_url}"));
                env.push(format!("http_proxy={proxy_url}"));
                env.push(format!("https_proxy={proxy_url}"));
                env.push("NO_PROXY=localhost,127.0.0.1".to_string());
                env.push("no_proxy=localhost,127.0.0.1".to_string());
                (Some(net_name), Some(sidecar))
            }
        };

        let host_config = self.build_host_config(&spec.mounts, network_mode);

        let config: Config<String> = Config {
            image: Some(self.policy.image.clone()),
            cmd: Some(spec.cmd.clone()),
            env: Some(env),
            working_dir: Some(spec.workdir.display().to_string()),
            attach_stdin: Some(spec.stdin.is_some()),
            attach_stdout: Some(true),
            attach_stderr: Some(true),
            open_stdin: Some(spec.stdin.is_some()),
            stdin_once: Some(spec.stdin.is_some()),
            tty: Some(false),
            host_config: Some(host_config),
            ..Default::default()
        };

        let name = format!("brainwires-sandbox-{}", handle.as_uuid());
        let create_opts = CreateContainerOptions {
            name: name.clone(),
            platform: None,
        };

        let created = match self
            .client
            .create_container(Some(create_opts), config)
            .await
        {
            Ok(c) => c,
            Err(e) => {
                if let Some(sidecar) = limited {
                    self.teardown_limited(sidecar).await;
                }
                return Err(e.into());
            }
        };
        let container_id = created.id;

        let mut attach = self
            .client
            .attach_container(
                &container_id,
                Some(AttachContainerOptions::<String> {
                    stdin: Some(spec.stdin.is_some()),
                    stdout: Some(true),
                    stderr: Some(true),
                    stream: Some(true),
                    logs: Some(true),
                    detach_keys: None,
                }),
            )
            .await?;

        self.client
            .start_container(&container_id, None::<StartContainerOptions<String>>)
            .await?;

        if let Some(bytes) = spec.stdin.as_ref() {
            use tokio::io::AsyncWriteExt;
            attach.input.write_all(bytes).await?;
            attach.input.shutdown().await?;
        }

        let job = Job {
            container_id,
            started: Instant::now(),
            timeout: spec.timeout,
            attach,
            limited,
        };
        self.jobs.lock().await.insert(handle, job);
        Ok(handle)
    }

    async fn wait(&self, handle: ExecHandle) -> Result<ExecOutput> {
        let Job {
            container_id,
            started,
            timeout: timeout_dur,
            mut attach,
            limited,
        } = self
            .jobs
            .lock()
            .await
            .remove(&handle)
            .ok_or_else(|| SandboxError::NotAvailable("unknown exec handle".into()))?;

        let client = self.client.clone();
        let collect_and_wait = async {
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            while let Some(frame) = attach.output.next().await {
                let frame = frame.map_err(|e| SandboxError::Docker(e.to_string()))?;
                match frame {
                    LogOutput::StdOut { message } => stdout.extend_from_slice(&message),
                    LogOutput::StdErr { message } => stderr.extend_from_slice(&message),
                    LogOutput::Console { message } => stdout.extend_from_slice(&message),
                    LogOutput::StdIn { .. } => {}
                }
            }

            let mut wait_stream = client.wait_container(
                &container_id,
                None::<bollard::container::WaitContainerOptions<String>>,
            );
            let mut exit_code: i64 = 0;
            while let Some(ev) = wait_stream.next().await {
                match ev {
                    Ok(resp) => exit_code = resp.status_code,
                    Err(bollard::errors::Error::DockerContainerWaitError { code, .. }) => {
                        exit_code = code;
                    }
                    Err(e) => return Err(SandboxError::Docker(e.to_string())),
                }
            }

            Ok::<_, SandboxError>(ExecOutput {
                exit_code: exit_code as i32,
                stdout,
                stderr,
                wall_time: started.elapsed(),
            })
        };

        let result = match timeout(timeout_dur, collect_and_wait).await {
            Ok(res) => res,
            Err(_) => {
                tracing::warn!(
                    container_id = %container_id,
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    timeout_ms = timeout_dur.as_millis() as u64,
                    "DockerSandbox exec exceeded ExecSpec.timeout; killing container"
                );
                let _ = self
                    .client
                    .kill_container(
                        &container_id,
                        None::<bollard::container::KillContainerOptions<String>>,
                    )
                    .await;
                Err(SandboxError::Timeout)
            }
        };

        let _ = self
            .client
            .remove_container(
                &container_id,
                Some(RemoveContainerOptions {
                    force: true,
                    v: true,
                    link: false,
                }),
            )
            .await;

        if let Some(sidecar) = limited {
            self.teardown_limited(sidecar).await;
        }

        result
    }

    async fn shutdown(&self) -> Result<()> {
        let jobs: Vec<_> = {
            let mut guard = self.jobs.lock().await;
            guard.drain().collect()
        };
        for (_, job) in jobs {
            let _ = self
                .client
                .remove_container(
                    &job.container_id,
                    Some(RemoveContainerOptions {
                        force: true,
                        v: true,
                        link: false,
                    }),
                )
                .await;
            if let Some(sidecar) = job.limited {
                self.teardown_limited(sidecar).await;
            }
        }
        Ok(())
    }

    fn runtime(&self) -> SandboxRuntime {
        self.policy.runtime
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn curl_spec(url: &str) -> ExecSpec {
        ExecSpec {
            cmd: vec![
                "curl".into(),
                "-sS".into(),
                "-o".into(),
                "/dev/null".into(),
                "-w".into(),
                "%{http_code}".into(),
                url.into(),
            ],
            env: BTreeMap::new(),
            workdir: PathBuf::from("/"),
            stdin: None,
            mounts: vec![],
            timeout: Duration::from_secs(30),
        }
    }

    #[tokio::test]
    #[ignore = "requires a live Docker daemon"]
    async fn echo_hello_in_docker() {
        let policy = SandboxPolicy {
            image: "alpine:3".into(),
            network: NetworkPolicy::None,
            ..SandboxPolicy::default()
        };
        let sandbox = DockerSandbox::connect(policy).expect("connect");

        let spec = ExecSpec {
            cmd: vec!["echo".into(), "hello".into()],
            env: BTreeMap::new(),
            workdir: PathBuf::from("/"),
            stdin: None,
            mounts: vec![],
            timeout: Duration::from_secs(30),
        };

        let handle = sandbox.spawn(spec).await.expect("spawn");
        let out = sandbox.wait(handle).await.expect("wait");
        assert_eq!(out.exit_code, 0);
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hello");
    }

    #[tokio::test]
    #[ignore = "requires live Docker + published proxy image"]
    async fn limited_network_allows_listed_host() {
        let policy = SandboxPolicy {
            image: "curlimages/curl:latest".into(),
            network: NetworkPolicy::Limited(vec!["pypi.org".into()]),
            read_only_rootfs: false,
            ..SandboxPolicy::default()
        };
        let sandbox = DockerSandbox::connect(policy).expect("connect");
        let handle = sandbox
            .spawn(curl_spec("https://pypi.org/"))
            .await
            .expect("spawn");
        let out = sandbox.wait(handle).await.expect("wait");
        assert_eq!(
            out.exit_code,
            0,
            "stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[tokio::test]
    #[ignore = "requires live Docker + published proxy image"]
    async fn limited_network_blocks_unlisted_host() {
        let policy = SandboxPolicy {
            image: "curlimages/curl:latest".into(),
            network: NetworkPolicy::Limited(vec!["pypi.org".into()]),
            read_only_rootfs: false,
            ..SandboxPolicy::default()
        };
        let sandbox = DockerSandbox::connect(policy).expect("connect");
        let handle = sandbox
            .spawn(curl_spec("https://example.com/"))
            .await
            .expect("spawn");
        let out = sandbox.wait(handle).await.expect("wait");
        assert_ne!(
            out.exit_code, 0,
            "expected curl to fail against blocked host"
        );
    }

    #[tokio::test]
    #[ignore = "requires live Docker + published proxy image"]
    async fn limited_network_blocks_raw_tcp() {
        let policy = SandboxPolicy {
            image: "busybox:latest".into(),
            network: NetworkPolicy::Limited(vec!["pypi.org".into()]),
            read_only_rootfs: false,
            ..SandboxPolicy::default()
        };
        let sandbox = DockerSandbox::connect(policy).expect("connect");
        let spec = ExecSpec {
            cmd: vec![
                "sh".into(),
                "-c".into(),
                "nc -w2 1.1.1.1 53 < /dev/null".into(),
            ],
            env: BTreeMap::new(),
            workdir: PathBuf::from("/"),
            stdin: None,
            mounts: vec![],
            timeout: Duration::from_secs(10),
        };
        let handle = sandbox.spawn(spec).await.expect("spawn");
        let out = sandbox.wait(handle).await.expect("wait");
        assert_ne!(
            out.exit_code, 0,
            "raw TCP should be blocked by internal network"
        );
    }
}
