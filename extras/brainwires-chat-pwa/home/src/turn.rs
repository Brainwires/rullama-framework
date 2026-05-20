//! Server-side minting of WebRTC ICE-server lists.
//!
//! When a Cloudflare Calls TURN config is supplied, mints a fresh short-lived
//! credential per session. Always falls back to (and includes alongside) public
//! STUN servers so connection-establishment doesn't hard-depend on the TURN
//! API being reachable.
//!
//! The PWA never holds the Cloudflare Calls API token — minting happens on the
//! daemon and the credentials are returned in the `POST /signal/session`
//! response with a ~10 minute lifetime. A new credential is minted per session.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::TurnConfig;

/// One ICE server entry, shaped to match the JS `RTCIceServer` dictionary
/// (https://w3c.github.io/webrtc-pc/#dom-rtciceserver). The PWA hands the
/// returned array directly to `new RTCPeerConnection({ iceServers: [...] })`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct IceServerJson {
    pub urls: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credential: Option<String>,
}

/// The Cloudflare TURN endpoint base. Overridable in tests via
/// [`mint_ice_servers_with_base`] so unit tests can point at a local stub.
pub const CF_TURN_BASE: &str = "https://rtc.live.cloudflare.com/v1/turn/keys";

/// Minimum credential TTL we'll request from Cloudflare. Anything shorter
/// than a minute is more likely to expire mid-handshake than to be useful.
pub const MIN_TTL_SECS: u32 = 60;

/// Default credential TTL if the caller didn't pick one. ~10 minutes:
/// long enough that a fresh session won't expire mid-handshake, short
/// enough that a leaked credential isn't useful for long.
pub const DEFAULT_TTL_SECS: u32 = 600;

/// Mint an array of `RTCIceServer`-shaped entries for a new signaling session.
///
/// On TURN-config presence: hits Cloudflare's API and returns
/// `[<cf turn entry>, <stun fallbacks...>]`. On absence (or transient API
/// failure): returns just the STUN fallbacks. Errors on TURN minting are
/// logged but do NOT fail the session — STUN-only is still a valid path.
pub async fn mint_ice_servers(cfg: &TurnConfig, http: &reqwest::Client) -> Vec<IceServerJson> {
    mint_ice_servers_with_base(cfg, http, CF_TURN_BASE).await
}

/// Same as [`mint_ice_servers`] but with an overridable Cloudflare base URL.
/// Used by tests to point at a local stub TCP listener.
pub async fn mint_ice_servers_with_base(
    cfg: &TurnConfig,
    http: &reqwest::Client,
    cf_base: &str,
) -> Vec<IceServerJson> {
    let mut out = Vec::new();
    if let (Some(key_id), Some(token)) = (cfg.turn_key_id.as_ref(), cfg.api_token.as_ref()) {
        let ttl = cfg.credential_ttl_secs.max(MIN_TTL_SECS);
        match mint_cloudflare(cf_base, key_id, token, ttl, http).await {
            Ok(srv) => out.push(srv),
            Err(e) => {
                tracing::warn!(error = %e, "TURN: cloudflare minting failed; STUN-only fallback")
            }
        }
    }
    out.push(stun_only("stun:stun.cloudflare.com:3478"));
    out.push(stun_only("stun:stun.l.google.com:19302"));
    out
}

fn stun_only(url: &str) -> IceServerJson {
    IceServerJson {
        urls: vec![url.to_string()],
        username: None,
        credential: None,
    }
}

async fn mint_cloudflare(
    cf_base: &str,
    turn_key_id: &str,
    api_token: &str,
    ttl: u32,
    http: &reqwest::Client,
) -> Result<IceServerJson> {
    #[derive(Serialize)]
    struct ReqBody {
        ttl: u32,
    }
    #[derive(Deserialize)]
    struct CfResp {
        #[serde(rename = "iceServers")]
        ice_servers: IceServerJson,
    }

    let url = format!(
        "{}/{}/credentials/generate-ice-servers",
        cf_base.trim_end_matches('/'),
        turn_key_id
    );
    let resp = http
        .post(&url)
        .bearer_auth(api_token)
        .json(&ReqBody { ttl })
        .send()
        .await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("cloudflare turn API {}: {}", status, body);
    }
    Ok(resp.json::<CfResp>().await?.ice_servers)
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Spin up a tiny HTTP/1.1 stub on `127.0.0.1:0` that returns `body`
    /// with `Content-Type: application/json` for the first request, then
    /// closes. Returns (`http://127.0.0.1:PORT`, request-count handle).
    ///
    /// Hand-rolled because `wiremock` isn't a workspace dep and this is
    /// the only place we mock HTTP — pulling it in for three tests would
    /// be heavy.
    async fn stub_server(
        status_line: &'static str,
        body: &'static str,
    ) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();
        tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(s) => s,
                    Err(_) => break,
                };
                let count_clone = count_clone.clone();
                tokio::spawn(async move {
                    // Drain the request — we don't validate it, just count it.
                    let mut buf = [0u8; 4096];
                    let _ = sock.read(&mut buf).await;
                    count_clone.fetch_add(1, Ordering::SeqCst);
                    let resp = format!(
                        "{status_line}\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len(),
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        (format!("http://{addr}"), count)
    }

    #[tokio::test]
    async fn mint_ice_servers_no_config_returns_stun_fallbacks() {
        let cfg = TurnConfig::default();
        let http = reqwest::Client::new();
        let out = mint_ice_servers(&cfg, &http).await;
        assert_eq!(
            out.len(),
            2,
            "STUN-only fallback should yield 2 entries: {out:?}"
        );
        assert_eq!(out[0].urls[0], "stun:stun.cloudflare.com:3478");
        assert_eq!(out[1].urls[0], "stun:stun.l.google.com:19302");
        for s in &out {
            assert!(s.username.is_none(), "STUN entries must not carry creds");
            assert!(s.credential.is_none(), "STUN entries must not carry creds");
        }
    }

    #[tokio::test]
    async fn mint_ice_servers_cf_failure_falls_back() {
        // Point at an unreachable port — connect refused, no panic.
        let cfg = TurnConfig {
            turn_key_id: Some("KEYID".into()),
            api_token: Some("tok".into()),
            credential_ttl_secs: 600,
        };
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()
            .unwrap();
        let out = mint_ice_servers_with_base(&cfg, &http, "http://127.0.0.1:1").await;
        assert_eq!(
            out.len(),
            2,
            "transient CF failure must NOT fail the session: {out:?}"
        );
        assert_eq!(out[0].urls[0], "stun:stun.cloudflare.com:3478");
    }

    #[tokio::test]
    async fn mint_ice_servers_cf_500_falls_back() {
        let (base, count) =
            stub_server("HTTP/1.1 500 Internal Server Error", "{\"error\":\"oops\"}").await;
        let cfg = TurnConfig {
            turn_key_id: Some("KEYID".into()),
            api_token: Some("tok".into()),
            credential_ttl_secs: 600,
        };
        let http = reqwest::Client::new();
        let out = mint_ice_servers_with_base(&cfg, &http, &base).await;
        assert_eq!(out.len(), 2, "5xx must fall back to STUN-only: {out:?}");
        assert!(
            count.load(Ordering::SeqCst) >= 1,
            "stub should have been hit"
        );
    }

    #[tokio::test]
    async fn mint_ice_servers_cf_success_includes_turn_entry() {
        let body = r#"{"iceServers":{"urls":["turn:turn.cloudflare.com:3478","turns:turn.cloudflare.com:5349"],"username":"u","credential":"c"}}"#;
        let (base, _count) = stub_server("HTTP/1.1 200 OK", body).await;
        let cfg = TurnConfig {
            turn_key_id: Some("KEYID".into()),
            api_token: Some("tok".into()),
            credential_ttl_secs: 600,
        };
        let http = reqwest::Client::new();
        let out = mint_ice_servers_with_base(&cfg, &http, &base).await;
        assert_eq!(out.len(), 3, "TURN + 2 STUN fallbacks expected: {out:?}");
        let turn = &out[0];
        assert_eq!(turn.username.as_deref(), Some("u"));
        assert_eq!(turn.credential.as_deref(), Some("c"));
        assert!(
            turn.urls.iter().any(|u| u.starts_with("turn:")),
            "TURN entry urls: {:?}",
            turn.urls
        );
        // STUN fallbacks are still appended.
        assert_eq!(out[1].urls[0], "stun:stun.cloudflare.com:3478");
        assert_eq!(out[2].urls[0], "stun:stun.l.google.com:19302");
    }

    #[test]
    fn ttl_floor_is_enforced() {
        // The .max(MIN_TTL_SECS) guard means even ttl=0 won't sneak past.
        let cfg = TurnConfig {
            turn_key_id: Some("k".into()),
            api_token: Some("t".into()),
            credential_ttl_secs: 0,
        };
        assert_eq!(cfg.credential_ttl_secs.max(MIN_TTL_SECS), MIN_TTL_SECS);
    }
}
