//! `brainclaw doctor` — diagnostic report across subsystems.
//!
//! Each check returns a [`CheckResult`] with a status (Pass, Warn, Fail,
//! Skip), a short human-readable detail string and an optional fix hint.
//! The report is rendered to stdout either as human text (coloured when
//! stdout is a TTY) or as a machine-readable JSON array.
//!
//! This module is intentionally self-contained: it only reads from the
//! `BrainClawConfig` and the process environment. It never mutates live
//! daemon state. Network I/O is bounded by short timeouts so `doctor`
//! stays fast even when the network is unreachable.
//!
//! The doctor is a diagnostic, NOT a live probe of a running daemon. It
//! reports on the configuration as written and does its own light
//! connectivity checks (e.g. a TCP connect on the gateway port).

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use clap::Args;
use serde::Serialize;

use crate::config::BrainClawConfig;

/// Outcome of a single diagnostic check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// The check succeeded outright.
    Pass,
    /// The check was partially successful or could not be completed
    /// definitively (e.g. network unavailable).
    Warn,
    /// The check definitively failed — the operator should act.
    Fail,
    /// The check was not applicable (e.g. feature disabled).
    Skip,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Warn => "WARN",
            Status::Fail => "FAIL",
            Status::Skip => "SKIP",
        }
    }

    fn ansi_color(self) -> &'static str {
        match self {
            Status::Pass => "\x1b[32m", // green
            Status::Warn => "\x1b[33m", // yellow
            Status::Fail => "\x1b[31m", // red
            Status::Skip => "\x1b[90m", // bright black (grey)
        }
    }
}

/// A single named diagnostic check.
#[derive(Debug, Clone, Serialize)]
pub struct CheckResult {
    /// Short human name of the check.
    pub name: String,
    /// Status bucket — dictates exit code and colour.
    pub status: Status,
    /// Human-readable detail about the outcome.
    pub detail: String,
    /// Optional fix hint the operator can follow.
    pub fix_hint: Option<String>,
}

impl CheckResult {
    fn pass(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: Status::Pass,
            detail: detail.into(),
            fix_hint: None,
        }
    }
    fn warn(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: Status::Warn,
            detail: detail.into(),
            fix_hint: None,
        }
    }
    fn fail(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: Status::Fail,
            detail: detail.into(),
            fix_hint: None,
        }
    }
    fn skip(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: Status::Skip,
            detail: detail.into(),
            fix_hint: None,
        }
    }
    fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.fix_hint = Some(hint.into());
        self
    }
}

/// Command-line flags for `brainclaw doctor`.
#[derive(Debug, Clone, Args, Default)]
pub struct DoctorArgs {
    /// Suppress `PASS` rows in human-readable output.
    #[arg(long)]
    pub quiet: bool,
    /// Output format: `text` (default) or `json`.
    #[arg(long, default_value = "text")]
    pub format: String,
}

/// Output format for the doctor report.
#[derive(Debug, Clone, Copy)]
enum OutputFormat {
    Text,
    Json,
}

impl OutputFormat {
    fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            other => anyhow::bail!("unknown --format value '{}' (expected text|json)", other),
        }
    }
}

/// Entry point wired from `main.rs`.
///
/// `config_override` is the value of the `--config` flag — used so the
/// Config-load check can report Skip vs Fail correctly.
pub async fn run(
    config_override: Option<&str>,
    args: &DoctorArgs,
) -> Result<(Vec<CheckResult>, i32)> {
    let format = OutputFormat::parse(&args.format)?;

    // 1. Try to load config — this is itself a check, and also drives
    //    every subsequent check. If load fails we still run the rest
    //    with defaults so the operator sees as much as possible.
    let (load_result, config, used_config_path): (CheckResult, BrainClawConfig, Option<PathBuf>) =
        match load_config_for_doctor(config_override) {
            Ok((cfg, path)) => {
                let detail = match &path {
                    Some(p) => format!("loaded {}", p.display()),
                    None => "using built-in defaults (no config file found)".to_string(),
                };
                let status = if config_override.is_some() && path.is_none() {
                    // Caller passed --config but it couldn't be read — already errored
                    // above; unreachable here. Keep as Pass for safety.
                    Status::Pass
                } else if path.is_some() {
                    Status::Pass
                } else {
                    Status::Skip
                };
                let mut r = match status {
                    Status::Pass => CheckResult::pass("config-load", detail),
                    _ => CheckResult::skip("config-load", detail),
                };
                if status == Status::Skip {
                    r = r.with_hint("run `brainclaw onboard` to generate a config file");
                }
                (r, cfg, path)
            }
            Err(e) => {
                // Fail config-load, but continue with defaults so the rest
                // of the checks still run.
                let r = CheckResult::fail("config-load", e.to_string())
                    .with_hint("fix the TOML parse error or pass --config <path>");
                (r, BrainClawConfig::default(), None)
            }
        };

    let mut results: Vec<CheckResult> = vec![load_result];

    results.extend(run_all_checks(&config, used_config_path.as_deref()).await);

    // 2. Render.
    render_report(&results, format, args.quiet)?;

    // 3. Exit code.
    let exit = if results.iter().any(|r| r.status == Status::Fail) {
        1
    } else {
        0
    };
    Ok((results, exit))
}

/// Run the full check battery against a supplied config.
///
/// Public so integration tests can exercise the pipeline without hitting
/// the network or the filesystem for the config file itself.
pub async fn run_doctor_with_config(config: &BrainClawConfig) -> Vec<CheckResult> {
    run_all_checks(config, None).await
}

async fn run_all_checks(config: &BrainClawConfig, _config_path: Option<&Path>) -> Vec<CheckResult> {
    let mut out = Vec::new();

    out.push(check_provider_auth(config).await);
    out.push(check_gateway_reachable(config).await);
    out.extend(check_sandbox(config).await);
    out.extend(check_channel_credentials(config));
    out.push(check_skill_registry(config).await);
    out.extend(check_skill_directories(config));
    out.push(check_pairing_store(config));
    out.push(check_memory(config));
    out.push(check_disk_space(config));
    out.push(check_ports(config).await);
    out.extend(check_gmail_push(config).await);

    out
}

/// `gmail-push` — one row per configured account.
///
/// Verifies that an OAuth token is reachable and, when reachable, asks
/// Gmail to register (or refresh) the watch. A successful `users.watch`
/// response is treated as Pass; a 4xx is Fail (token/scope problem);
/// anything else is Warn.
///
/// The OAuth refresh path is not implemented here — if the token is
/// expired, Gmail responds 401 and this check reports Fail. That is the
/// correct and intended behaviour for now: operators must supply a
/// fresh token (typically from the BrainClaw OAuth skill or a paired
/// Google Cloud service account).
async fn check_gmail_push(config: &BrainClawConfig) -> Vec<CheckResult> {
    if !config.gmail_push.enabled {
        return vec![CheckResult::skip(
            "gmail-push",
            "gmail_push.enabled = false",
        )];
    }
    if config.gmail_push.accounts.is_empty() {
        return vec![CheckResult::warn(
            "gmail-push",
            "gmail_push.enabled but no accounts configured",
        )];
    }

    #[cfg(not(feature = "email-push"))]
    {
        return vec![CheckResult::warn(
            "gmail-push",
            "daemon built without the `email-push` feature; Gmail push is disabled at runtime",
        )];
    }

    #[cfg(feature = "email-push")]
    {
        use brainwires_tools::gmail_push::{GmailPushConfig, GmailPushHandler};

        let mut out = Vec::new();
        for acct in &config.gmail_push.accounts {
            let name = format!("gmail-push:{}", acct.email_address);
            let token = match (&acct.oauth_token_env, &acct.oauth_token) {
                (Some(env), _) => match std::env::var(env) {
                    Ok(v) if !v.is_empty() => v,
                    _ => {
                        out.push(
                            CheckResult::fail(
                                name.clone(),
                                format!("oauth_token_env '{env}' is not set"),
                            )
                            .with_hint(format!("export {env}=... before starting the daemon")),
                        );
                        continue;
                    }
                },
                (None, Some(inline)) if !inline.is_empty() => inline.clone(),
                _ => {
                    out.push(
                        CheckResult::fail(name.clone(), "no oauth_token_env or oauth_token set")
                            .with_hint("set gmail_push.accounts.[..].oauth_token_env"),
                    );
                    continue;
                }
            };

            let cfg = GmailPushConfig {
                project_id: acct.project_id.clone(),
                topic_name: acct.topic_name.clone(),
                push_audience: acct.push_audience.clone(),
                watched_label_ids: if acct.watched_label_ids.is_empty() {
                    vec!["INBOX".to_string()]
                } else {
                    acct.watched_label_ids.clone()
                },
                oauth_token: token,
                gmail_base_url: None,
            };
            let handler = GmailPushHandler::new(cfg);
            // Bound the network call so `doctor` stays fast.
            match tokio::time::timeout(Duration::from_secs(10), handler.register_watch()).await {
                Ok(Ok(resp)) => out.push(CheckResult::pass(
                    name,
                    format!(
                        "watch registered (history_id={}, expires {})",
                        resp.history_id, resp.expiration
                    ),
                )),
                Ok(Err(e)) => {
                    let msg = e.to_string();
                    let is_auth = msg.contains("401") || msg.contains("403");
                    let r = if is_auth {
                        CheckResult::fail(name, format!("users.watch rejected: {msg}"))
                            .with_hint("refresh the OAuth token — Gmail rejected the current one")
                    } else {
                        CheckResult::warn(name, format!("users.watch error: {msg}"))
                    };
                    out.push(r);
                }
                Err(_) => out.push(CheckResult::warn(
                    name,
                    "users.watch timed out after 10s — network slow or offline",
                )),
            }
        }
        out
    }
}

// ── Check implementations ───────────────────────────────────────────────

fn load_config_for_doctor(
    config_override: Option<&str>,
) -> Result<(BrainClawConfig, Option<PathBuf>)> {
    if let Some(p) = config_override {
        let path = PathBuf::from(p);
        let cfg = BrainClawConfig::load(&path)?;
        return Ok((cfg, Some(path)));
    }
    // Replicate load_or_default's search order but track which path matched.
    if let Some(home) = dirs::home_dir() {
        let home_cfg = home.join(".brainclaw").join("brainclaw.toml");
        if home_cfg.exists() {
            let cfg = BrainClawConfig::load(&home_cfg)?;
            return Ok((cfg, Some(home_cfg)));
        }
    }
    let local = PathBuf::from("brainclaw.toml");
    if local.exists() {
        let cfg = BrainClawConfig::load(&local)?;
        return Ok((cfg, Some(local)));
    }
    Ok((BrainClawConfig::default(), None))
}

async fn check_provider_auth(config: &BrainClawConfig) -> CheckResult {
    let provider = config.provider.default_provider.as_str();
    let (default_env, probe_url): (&'static str, Option<&'static str>) = match provider {
        "anthropic" => (
            "ANTHROPIC_API_KEY",
            Some("https://api.anthropic.com/v1/models"),
        ),
        "openai" | "openai-responses" | "openai_responses" => {
            ("OPENAI_API_KEY", Some("https://api.openai.com/v1/models"))
        }
        "google" | "gemini" => ("GOOGLE_API_KEY", None),
        "groq" => (
            "GROQ_API_KEY",
            Some("https://api.groq.com/openai/v1/models"),
        ),
        "ollama" => ("", None),
        "together" => ("TOGETHER_API_KEY", None),
        "fireworks" => ("FIREWORKS_API_KEY", None),
        "anyscale" => ("ANYSCALE_API_KEY", None),
        "brainwires" => ("BRAINWIRES_API_KEY", None),
        _ => ("", None),
    };

    // Ollama doesn't need a key.
    if provider == "ollama" {
        return CheckResult::pass("provider-auth", "ollama configured — no API key required");
    }

    // Resolve the key.
    let key = resolve_provider_key(config, default_env);

    let Some(key) = key else {
        let msg = format!(
            "no API key found for provider '{provider}' (checked config.provider.api_key, \
             api_key_env, and env var '{default_env}')"
        );
        return CheckResult::fail("provider-auth", msg).with_hint(format!(
            "export {default_env}=... or set provider.api_key_env"
        ));
    };

    // Network probe — bounded to 5s.
    let Some(url) = probe_url else {
        return CheckResult::warn(
            "provider-auth",
            format!("API key present for '{provider}' (no lightweight probe implemented)"),
        );
    };

    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return CheckResult::warn(
                "provider-auth",
                format!("couldn't construct HTTP client: {e}"),
            );
        }
    };
    let req = match provider {
        "anthropic" => client
            .get(url)
            .header("x-api-key", &key)
            .header("anthropic-version", "2023-06-01"),
        _ => client.get(url).bearer_auth(&key),
    };
    match req.send().await {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                CheckResult::pass(
                    "provider-auth",
                    format!("{provider} models endpoint returned {status}"),
                )
            } else if status.as_u16() == 401 || status.as_u16() == 403 {
                CheckResult::fail(
                    "provider-auth",
                    format!("{provider} rejected the API key ({status})"),
                )
                .with_hint("check the key is valid and has the right scopes")
            } else {
                CheckResult::warn("provider-auth", format!("{provider} returned {status}"))
            }
        }
        Err(e) => CheckResult::warn(
            "provider-auth",
            format!("couldn't reach {provider} ({e}) — network may be offline"),
        ),
    }
}

fn resolve_provider_key(config: &BrainClawConfig, default_env: &str) -> Option<String> {
    if let Some(ref k) = config.provider.api_key
        && !k.is_empty()
    {
        return Some(k.clone());
    }
    if let Some(ref env_name) = config.provider.api_key_env
        && let Ok(v) = std::env::var(env_name)
        && !v.is_empty()
    {
        return Some(v);
    }
    if !default_env.is_empty()
        && let Ok(v) = std::env::var(default_env)
        && !v.is_empty()
    {
        return Some(v);
    }
    None
}

async fn check_gateway_reachable(config: &BrainClawConfig) -> CheckResult {
    use tokio::net::TcpStream;
    let host = if config.gateway.host == "0.0.0.0" {
        "127.0.0.1".to_string()
    } else {
        config.gateway.host.clone()
    };
    let addr = format!("{host}:{}", config.gateway.port);
    match tokio::time::timeout(Duration::from_secs(1), TcpStream::connect(&addr)).await {
        Ok(Ok(_)) => CheckResult::pass(
            "gateway-reachable",
            format!("TCP connected to {addr} — a daemon is listening"),
        ),
        Ok(Err(e)) => {
            if e.kind() == std::io::ErrorKind::ConnectionRefused {
                CheckResult::warn(
                    "gateway-reachable",
                    format!("no gateway listening on {addr}"),
                )
                .with_hint("run `brainclaw serve` to start the daemon")
            } else {
                CheckResult::fail(
                    "gateway-reachable",
                    format!("TCP connect to {addr} failed: {e}"),
                )
            }
        }
        Err(_) => CheckResult::warn(
            "gateway-reachable",
            format!("TCP connect to {addr} timed out after 1s"),
        ),
    }
}

async fn check_sandbox(config: &BrainClawConfig) -> Vec<CheckResult> {
    let sb = &config.sandbox;
    if !sb.enabled {
        return vec![CheckResult::skip(
            "sandbox",
            "sandbox disabled in config.sandbox.enabled",
        )];
    }

    let mut out = Vec::new();
    let runtime = sb.runtime.to_ascii_lowercase();

    match runtime.as_str() {
        "host" => {
            out.push(
                CheckResult::warn(
                    "sandbox-runtime",
                    "runtime = 'host' — NO isolation is applied; dev/testing only",
                )
                .with_hint("switch to runtime = 'docker' for production"),
            );
        }
        "docker" | "podman" => {
            out.extend(run_sandbox_runtime_checks(config, &runtime).await);
        }
        other => {
            out.push(
                CheckResult::fail(
                    "sandbox-runtime",
                    format!("unknown runtime '{other}' (expected 'docker', 'podman', or 'host')"),
                )
                .with_hint("set sandbox.runtime = \"docker\""),
            );
        }
    }

    // Limited network policy sanity check (runtime-agnostic).
    if sb.network.eq_ignore_ascii_case("limited") && sb.allowed_hosts.is_empty() {
        out.push(
            CheckResult::warn(
                "sandbox-allowed-hosts",
                "network = 'limited' but allowed_hosts is empty — the allowlist blocks all egress",
            )
            .with_hint("add hostnames to sandbox.allowed_hosts"),
        );
    }

    out
}

#[cfg(feature = "sandbox")]
async fn run_sandbox_runtime_checks(config: &BrainClawConfig, runtime: &str) -> Vec<CheckResult> {
    use brainwires_sandbox::{DockerSandbox, ExecSpec, Sandbox};
    use std::collections::BTreeMap;

    let mut out = Vec::new();

    let policy = match config.sandbox.to_policy() {
        Ok(p) => p,
        Err(e) => {
            out.push(CheckResult::fail(
                "sandbox-policy",
                format!("config.sandbox -> policy: {e}"),
            ));
            return out;
        }
    };

    let sandbox = match DockerSandbox::connect(policy.clone()) {
        Ok(s) => s,
        Err(e) => {
            let hint = if runtime == "podman" {
                "ensure podman.socket is running and PODMAN_SOCKET or DOCKER_HOST is set"
            } else {
                "ensure docker is running and the current user has socket access"
            };
            out.push(
                CheckResult::fail("sandbox-runtime", format!("{runtime}: connect failed: {e}"))
                    .with_hint(hint),
            );
            return out;
        }
    };

    out.push(CheckResult::pass(
        "sandbox-runtime",
        format!("{runtime}: connect OK (image = {})", policy.image),
    ));

    // End-to-end spawn + wait (10s budget).
    let spec = ExecSpec {
        cmd: vec!["echo".into(), "hi".into()],
        env: BTreeMap::new(),
        workdir: std::path::PathBuf::from("/"),
        stdin: None,
        mounts: vec![],
        timeout: Duration::from_secs(10),
    };
    let probe = tokio::time::timeout(Duration::from_secs(10), async {
        let handle = sandbox.spawn(spec).await?;
        sandbox.wait(handle).await
    })
    .await;

    match probe {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stdout_trim = stdout.trim();
            if output.exit_code == 0 && stdout_trim == "hi" {
                out.push(CheckResult::pass(
                    "sandbox-echo",
                    "spawn(echo hi) returned exit=0 stdout=hi",
                ));
            } else {
                out.push(
                    CheckResult::fail(
                        "sandbox-echo",
                        format!(
                            "spawn(echo hi) returned exit={} stdout={stdout_trim:?}",
                            output.exit_code
                        ),
                    )
                    .with_hint("verify the sandbox image has a working busybox/coreutils"),
                );
            }
        }
        Ok(Err(e)) => {
            out.push(
                CheckResult::fail("sandbox-echo", format!("spawn failed: {e}"))
                    .with_hint("check the sandbox image is pulled and the runtime is healthy"),
            );
        }
        Err(_) => {
            out.push(
                CheckResult::fail("sandbox-echo", "spawn(echo hi) timed out after 10s")
                    .with_hint("pull the sandbox image ahead of time"),
            );
        }
    }

    out
}

#[cfg(not(feature = "sandbox"))]
async fn run_sandbox_runtime_checks(_config: &BrainClawConfig, runtime: &str) -> Vec<CheckResult> {
    vec![CheckResult::warn(
        "sandbox-runtime",
        format!("runtime = '{runtime}' but brainclaw was built without the `sandbox` feature"),
    )]
}

fn check_channel_credentials(config: &BrainClawConfig) -> Vec<CheckResult> {
    // BrainClaw channel adapters live in separate mcp-* crates and read
    // their credentials from env vars. The daemon's own config carries an
    // `allowed_channel_types` list, which we treat as the operator's
    // intent to enable each channel. When empty, channels are not
    // configured explicitly and there is nothing to check.
    let channels = &config.security.allowed_channel_types;
    if channels.is_empty() {
        return vec![CheckResult::skip(
            "channel-credentials",
            "no channels listed in security.allowed_channel_types",
        )];
    }
    let mut out = Vec::new();
    for ch in channels {
        let name = format!("channel-{}", ch.to_ascii_lowercase());
        // Some channels require multiple env vars — handle each group.
        match ch.to_ascii_lowercase().as_str() {
            "google_chat" | "google-chat" | "googlechat" => {
                out.extend(check_multi_env(
                    &name,
                    &[
                        "GOOGLE_CHAT_PROJECT_ID",
                        "GOOGLE_CHAT_AUDIENCE",
                        "GOOGLE_CHAT_SERVICE_ACCOUNT_KEY",
                    ],
                ));
                continue;
            }
            "teams" | "ms_teams" | "msteams" => {
                out.extend(check_multi_env(
                    &name,
                    &["TEAMS_APP_ID", "TEAMS_APP_PASSWORD", "TEAMS_TENANT_ID"],
                ));
                continue;
            }
            "irc" => {
                out.extend(check_multi_env(&name, &["IRC_SERVER", "IRC_NICK"]));
                continue;
            }
            "imessage" | "bluebubbles" => {
                out.extend(check_multi_env(&name, &["BB_SERVER_URL", "BB_PASSWORD"]));
                continue;
            }
            "nextcloud_talk" | "nextcloud-talk" | "nextcloudtalk" => {
                out.extend(check_multi_env(
                    &name,
                    &[
                        "NEXTCLOUD_URL",
                        "NEXTCLOUD_USERNAME",
                        "NEXTCLOUD_APP_PASSWORD",
                        "NEXTCLOUD_ROOMS",
                    ],
                ));
                continue;
            }
            "line" => {
                out.extend(check_multi_env(
                    &name,
                    &["LINE_CHANNEL_SECRET", "LINE_CHANNEL_ACCESS_TOKEN"],
                ));
                continue;
            }
            "feishu" | "lark" => {
                out.extend(check_multi_env(
                    &name,
                    &[
                        "FEISHU_APP_ID",
                        "FEISHU_APP_SECRET",
                        "FEISHU_VERIFICATION_TOKEN",
                    ],
                ));
                continue;
            }
            _ => {}
        }
        let env_var = match ch.to_ascii_lowercase().as_str() {
            "discord" => "DISCORD_TOKEN",
            "telegram" => "TELEGRAM_BOT_TOKEN",
            "slack" => "SLACK_BOT_TOKEN",
            "matrix" => "MATRIX_ACCESS_TOKEN",
            "mattermost" => "MATTERMOST_TOKEN",
            "whatsapp" => "WHATSAPP_ACCESS_TOKEN",
            "signal" => "SIGNAL_PHONE_NUMBER",
            "github" => "GITHUB_TOKEN",
            _ => {
                out.push(CheckResult::skip(
                    name,
                    format!("no known credential env var for channel '{ch}'"),
                ));
                continue;
            }
        };
        match std::env::var(env_var) {
            Ok(v) if !v.is_empty() => {
                out.push(CheckResult::pass(name, format!("{env_var} is set")))
            }
            _ => out.push(
                CheckResult::fail(name, format!("{env_var} is not set"))
                    .with_hint(format!("export {env_var}=... before starting the daemon")),
            ),
        }
    }
    out
}

/// Check that a group of env vars are all set; returns one `CheckResult`
/// per variable so the operator can see exactly which ones are missing.
fn check_multi_env(base_name: &str, vars: &[&str]) -> Vec<CheckResult> {
    let mut out = Vec::new();
    for v in vars {
        let name = format!("{base_name}:{v}");
        match std::env::var(v) {
            Ok(val) if !val.is_empty() => {
                out.push(CheckResult::pass(name, format!("{v} is set")));
            }
            _ => out.push(
                CheckResult::fail(name, format!("{v} is not set"))
                    .with_hint(format!("export {v}=... before starting the adapter")),
            ),
        }
    }
    out
}

async fn check_skill_registry(config: &BrainClawConfig) -> CheckResult {
    let Some(ref url) = config.skills.registry_url else {
        return CheckResult::skip("skill-registry", "config.skills.registry_url not set");
    };
    let probe = format!("{}/api/skills/list?limit=1", url.trim_end_matches('/'));
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return CheckResult::warn("skill-registry", format!("couldn't build HTTP client: {e}"));
        }
    };
    match client.get(&probe).send().await {
        Ok(r) if r.status().is_success() => {
            CheckResult::pass("skill-registry", format!("{probe} -> {}", r.status()))
        }
        Ok(r) => CheckResult::warn("skill-registry", format!("{probe} returned {}", r.status())),
        Err(e) => CheckResult::warn("skill-registry", format!("{probe} unreachable: {e}")),
    }
}

fn check_skill_directories(config: &BrainClawConfig) -> Vec<CheckResult> {
    if !config.skills.enabled {
        return vec![CheckResult::skip(
            "skill-directories",
            "skills.enabled = false",
        )];
    }
    if config.skills.directories.is_empty() {
        return vec![CheckResult::warn(
            "skill-directories",
            "skills.enabled but skills.directories is empty",
        )];
    }
    let mut out = Vec::new();
    for dir in &config.skills.directories {
        let path = expand_tilde(dir);
        let name = format!("skill-dir:{dir}");
        if !path.exists() {
            out.push(
                CheckResult::warn(name, format!("{} does not exist", path.display()))
                    .with_hint("create the directory or remove it from skills.directories"),
            );
            continue;
        }
        let count = count_skill_md(&path);
        out.push(CheckResult::pass(
            name,
            format!("{} ({count} SKILL.md files)", path.display()),
        ));
    }
    out
}

fn count_skill_md(dir: &Path) -> usize {
    fn walk(dir: &Path, acc: &mut usize, depth: usize) {
        if depth > 6 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, acc, depth + 1);
            } else if path.file_name().and_then(|n| n.to_str()) == Some("SKILL.md") {
                *acc += 1;
            }
        }
    }
    let mut n = 0;
    walk(dir, &mut n, 0);
    n
}

fn check_pairing_store(config: &BrainClawConfig) -> CheckResult {
    use brainwires_gateway::pairing::PairingStore;

    let path = match config.pairing.store_path.clone() {
        Some(p) => PathBuf::from(expand_tilde_str(&p)),
        None => match dirs::home_dir() {
            Some(h) => h.join(".brainclaw").join("pairing.json"),
            None => {
                return CheckResult::fail(
                    "pairing-store",
                    "HOME is not set and pairing.store_path is unconfigured",
                );
            }
        },
    };
    // PairingStore::load spawns a tokio task for the writer; we need a runtime.
    // The caller (`brainclaw doctor`) always runs inside #[tokio::main], but
    // `run_doctor_with_config` is also callable from tests — those must use
    // `#[tokio::test]`.
    match PairingStore::load(&path) {
        Ok(store) => {
            // Approved / pending counts are behind async locks; use
            // block_in_place isn't appropriate here because we aren't
            // guaranteed a multi-thread runtime. Do a blocking poll via
            // tokio::runtime::Handle::current and block_on from a separate
            // thread isn't portable either. The simplest correct thing is
            // to read the file directly, since the store path is a JSON
            // dump of `PairingState`.
            let raw = std::fs::read_to_string(&path).unwrap_or_default();
            let (approved, pending) = if raw.trim().is_empty() {
                (0usize, 0usize)
            } else {
                let v: serde_json::Value =
                    serde_json::from_str(&raw).unwrap_or(serde_json::Value::Null);
                let approved = v
                    .get("approved")
                    .and_then(|a| a.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);
                let pending = v
                    .get("pending")
                    .and_then(|p| p.as_object())
                    .map(|m| m.len())
                    .unwrap_or(0);
                (approved, pending)
            };
            let _ = store; // store drops, background writer exits cleanly.
            CheckResult::pass(
                "pairing-store",
                format!(
                    "{} ({approved} approved peers, {pending} pending codes)",
                    path.display()
                ),
            )
        }
        Err(e) => CheckResult::fail(
            "pairing-store",
            format!("failed to load {}: {e}", path.display()),
        )
        .with_hint("check permissions on the pairing store path"),
    }
}

fn check_memory(config: &BrainClawConfig) -> CheckResult {
    if !config.memory.enabled {
        return CheckResult::skip("memory", "memory.enabled = false");
    }
    let path = expand_tilde(&config.memory.storage_dir);
    if !path.exists() {
        // Not yet created — that's OK as long as the parent is writable.
        let parent = path.parent().unwrap_or(Path::new("."));
        if parent_writable(parent) {
            return CheckResult::pass(
                "memory",
                format!("{} (will be created on first use)", path.display()),
            );
        }
        return CheckResult::fail(
            "memory",
            format!(
                "{} does not exist and parent {} is not writable",
                path.display(),
                parent.display()
            ),
        );
    }
    if !path.is_dir() {
        return CheckResult::fail(
            "memory",
            format!("{} exists but is not a directory", path.display()),
        );
    }
    if dir_writable(&path) {
        CheckResult::pass("memory", format!("{} is writable", path.display()))
    } else {
        CheckResult::fail("memory", format!("{} is not writable", path.display()))
    }
}

fn parent_writable(parent: &Path) -> bool {
    if !parent.exists() {
        return false;
    }
    dir_writable(parent)
}

fn dir_writable(path: &Path) -> bool {
    // Best-effort: try to create a .brainclaw-doctor-probe file.
    let probe = path.join(".brainclaw-doctor-probe");
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

fn check_disk_space(config: &BrainClawConfig) -> CheckResult {
    // Cross-platform without new crates: use a coarse `statvfs`-like
    // heuristic. We avoid adding `sysinfo` here; instead try to create a
    // small file and detect ENOSPC — if it succeeds, we report "OK".
    // For "at least 500 MB free", we use `std::fs::metadata` unavailable.
    // We fall back to a best-effort warning instead of claiming a number
    // we haven't verified.
    let target = if config.memory.enabled {
        expand_tilde(&config.memory.storage_dir)
    } else if !config.skills.directories.is_empty() {
        expand_tilde(&config.skills.directories[0])
    } else {
        return CheckResult::skip("disk-space", "no memory/skills path to check");
    };
    let probe_dir = target
        .ancestors()
        .find(|p| p.exists())
        .unwrap_or(Path::new("/"));
    // Write 1MB; if that fails, the disk is either full or unwritable.
    let probe = probe_dir.join(".brainclaw-doctor-space-probe");
    let data = vec![0u8; 1024 * 1024];
    match std::fs::write(&probe, &data) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            // We didn't check the 500MB threshold — flag as Pass with a
            // caveat so the operator knows. Returning Warn on every host
            // would be too noisy.
            CheckResult::pass(
                "disk-space",
                format!(
                    "{} is writable (1 MB probe succeeded — coarse check)",
                    probe_dir.display()
                ),
            )
        }
        Err(e) => CheckResult::warn(
            "disk-space",
            format!(
                "couldn't write 1 MB probe under {}: {e}",
                probe_dir.display()
            ),
        ),
    }
}

async fn check_ports(config: &BrainClawConfig) -> CheckResult {
    use tokio::net::{TcpListener, TcpStream};
    let host = config.gateway.host.clone();
    let addr = format!("{host}:{}", config.gateway.port);

    // Try to bind first — if we succeed, the port is free, which means no
    // daemon is running there yet.
    match TcpListener::bind(&addr).await {
        Ok(_) => CheckResult::pass(
            "ports",
            format!("{addr} is bindable (no daemon currently running there)"),
        ),
        Err(bind_err) => {
            // Port is occupied. Probe: is it a brainclaw gateway?
            let connect_host = if host == "0.0.0.0" {
                "127.0.0.1"
            } else {
                host.as_str()
            };
            let probe_addr = format!("{connect_host}:{}", config.gateway.port);
            let ws =
                tokio::time::timeout(Duration::from_secs(1), TcpStream::connect(&probe_addr)).await;
            match ws {
                Ok(Ok(_)) => {
                    // Something is listening. Fire a quick HTTP probe to
                    // see if it responds like our gateway.
                    let url = format!(
                        "http://{connect_host}:{}/admin/metrics",
                        config.gateway.port
                    );
                    let client = reqwest::Client::builder()
                        .timeout(Duration::from_secs(1))
                        .build();
                    let is_gateway = match client {
                        Ok(c) => matches!(
                            c.get(&url).send().await,
                            Ok(r) if r.status().as_u16() < 500
                        ),
                        Err(_) => false,
                    };
                    if is_gateway {
                        CheckResult::pass(
                            "ports",
                            format!(
                                "{addr} occupied — brainclaw daemon appears to be running there"
                            ),
                        )
                    } else {
                        CheckResult::fail(
                            "ports",
                            format!(
                                "{addr} is taken by another process (no brainclaw admin API response)"
                            ),
                        )
                        .with_hint("pick a different gateway.port or stop the conflicting process")
                    }
                }
                _ => CheckResult::fail("ports", format!("{addr} bind failed: {bind_err}")),
            }
        }
    }
}

// ── Rendering ───────────────────────────────────────────────────────────

fn render_report(results: &[CheckResult], format: OutputFormat, quiet: bool) -> Result<()> {
    match format {
        OutputFormat::Json => {
            let s = serde_json::to_string_pretty(results)?;
            let mut out = std::io::stdout().lock();
            writeln!(out, "{s}")?;
            out.flush()?;
        }
        OutputFormat::Text => {
            let mut out = std::io::stdout().lock();
            let tty = std::io::stdout().is_terminal();
            let mut pass = 0usize;
            let mut warn = 0usize;
            let mut fail = 0usize;
            let mut skip = 0usize;
            for r in results {
                match r.status {
                    Status::Pass => pass += 1,
                    Status::Warn => warn += 1,
                    Status::Fail => fail += 1,
                    Status::Skip => skip += 1,
                }
                if quiet && r.status == Status::Pass {
                    continue;
                }
                if tty {
                    writeln!(
                        out,
                        "{color}[{label}]\x1b[0m {name}: {detail}",
                        color = r.status.ansi_color(),
                        label = r.status.label(),
                        name = r.name,
                        detail = r.detail,
                    )?;
                } else {
                    writeln!(
                        out,
                        "[{label}] {name}: {detail}",
                        label = r.status.label(),
                        name = r.name,
                        detail = r.detail,
                    )?;
                }
                if let Some(ref hint) = r.fix_hint {
                    writeln!(out, "        hint: {hint}")?;
                }
            }
            writeln!(
                out,
                "\nsummary: {pass} pass, {warn} warn, {fail} fail, {skip} skip"
            )?;
            out.flush()?;
        }
    }
    Ok(())
}

// ── Path helpers (local copies, to avoid dragging private helpers across) ─

fn expand_tilde(path: &str) -> PathBuf {
    PathBuf::from(expand_tilde_str(path))
}

fn expand_tilde_str(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest).to_string_lossy().into_owned();
    }
    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn provider_auth_fails_when_env_missing() {
        // Skip when the ambient environment has a key — we can't safely
        // remove env vars in parallel tests without race warnings.
        if std::env::var("ANTHROPIC_API_KEY").is_ok()
            || std::env::var("__BRAINCLAW_DOCTOR_TEST_MISSING_KEY__").is_ok()
        {
            eprintln!("skipping: required env vars happen to be set");
            return;
        }
        let mut cfg = BrainClawConfig::default();
        cfg.provider.default_provider = "anthropic".into();
        cfg.provider.api_key = None;
        cfg.provider.api_key_env = Some("__BRAINCLAW_DOCTOR_TEST_MISSING_KEY__".into());
        let r = check_provider_auth(&cfg).await;
        assert_eq!(r.status, Status::Fail);
    }

    #[tokio::test]
    async fn ollama_needs_no_key() {
        let mut cfg = BrainClawConfig::default();
        cfg.provider.default_provider = "ollama".into();
        let r = check_provider_auth(&cfg).await;
        assert_eq!(r.status, Status::Pass);
    }

    #[test]
    fn channel_check_skips_when_empty() {
        let cfg = BrainClawConfig::default();
        let r = check_channel_credentials(&cfg);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].status, Status::Skip);
    }

    #[test]
    fn new_channel_types_expand_to_multi_env_rows() {
        let mut cfg = BrainClawConfig::default();
        cfg.security.allowed_channel_types =
            vec!["google_chat".into(), "teams".into(), "irc".into()];
        let rows = check_channel_credentials(&cfg);
        // google_chat has 3 vars, teams has 3, irc has 2 → 8 rows.
        assert_eq!(rows.len(), 8);
        // Every row name must start with the adapter's prefix.
        assert!(rows.iter().any(|r| r.name.contains("google_chat")));
        assert!(rows.iter().any(|r| r.name.contains("teams")));
        assert!(rows.iter().any(|r| r.name.contains("irc")));
    }

    #[test]
    fn batch2_channel_types_expand_to_multi_env_rows() {
        let mut cfg = BrainClawConfig::default();
        cfg.security.allowed_channel_types = vec![
            "imessage".into(),
            "nextcloud_talk".into(),
            "line".into(),
            "feishu".into(),
        ];
        let rows = check_channel_credentials(&cfg);
        // imessage: 2, nextcloud_talk: 4, line: 2, feishu: 3 → 11
        assert_eq!(rows.len(), 11);
        assert!(rows.iter().any(|r| r.name.contains("imessage")));
        assert!(rows.iter().any(|r| r.name.contains("nextcloud_talk")));
        assert!(rows.iter().any(|r| r.name.contains("line")));
        assert!(rows.iter().any(|r| r.name.contains("feishu")));
    }
}
