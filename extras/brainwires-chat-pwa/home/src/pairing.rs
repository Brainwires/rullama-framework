//! Pairing endpoints — `/pair/claim` + `/pair/confirm` (Phase 2 M8).
//!
//! Flow:
//!   1. Operator runs `brainwires-home pair`. The daemon mints a random
//!      `one_time_token` + 6-digit `confirm_code`, prints a `bwhome://pair?...`
//!      URL (a QR-encodable URI) and the 6-digit code on stdout.
//!   2. PWA scans / pastes the URL, extracts `t=<one_time_token>`, POSTs
//!      `{one_time_token, device_pubkey, device_name}` → `/pair/claim`.
//!   3. PWA prompts the user for the 6-digit code, POSTs
//!      `{one_time_token, code}` → `/pair/confirm`.
//!   4. On match, the daemon mints a 32-byte hex `device_token`, appends a
//!      [`DeviceRecord`] to `~/.brainwires/home/devices.json` (atomic write,
//!      `0600` on Unix), drops the pending pair, and returns
//!      `{device_token, cf_client_id?, cf_client_secret?, peer_pubkey}`.
//!
//! ### CF Access tokens
//!
//! CF Zero Trust service-token minting via the Cloudflare API is **not**
//! implemented in M8 — the operator can configure a single pre-provisioned
//! `(cf_client_id, cf_client_secret)` pair via [`CfAccessConfig`], and the
//! daemon hands them to every PWA that successfully pairs. If the operator
//! hasn't configured CF Access at all, only the `device_token` is returned.
//!
//! ### Auth model
//!
//! The `device_token` is the daemon's primary auth gate: it's what the
//! signaling routes will validate on every request (M9 wires that). The CF
//! Access service-token pair is optional defence-in-depth at the tunnel
//! edge — a leaked CF token alone cannot reach the agent.
//!
//! ### `peer_pubkey`
//!
//! The plan calls for an Ed25519 peer fingerprint. M8 v1 ships a simpler
//! variant: a stable, randomly-generated 32-byte hex string the daemon
//! mints once at first start and persists at
//! `~/.brainwires/home/identity.json`. That's enough to anchor TOFU
//! ("trust on first use") on the PWA side; M11+ can swap in real Ed25519
//! signatures over signaling without breaking the wire format.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};
use chrono::Utc;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// How long a pending offer is honoured before it expires. After this the
/// PWA gets a 404 from `/pair/claim` (and `/pair/confirm`).
pub const OFFER_TTL: Duration = Duration::from_secs(5 * 60);

/// Length of the random one-time-token in bytes (hex-encoded → 64 chars).
pub const TOKEN_BYTES: usize = 32;

/// Length of the device_token in bytes (hex-encoded → 64 chars).
pub const DEVICE_TOKEN_BYTES: usize = 32;

/// Length of the home daemon's stable peer pubkey in bytes.
pub const PEER_PUBKEY_BYTES: usize = 32;

/// Pre-provisioned Cloudflare Access service-token pair. The operator sets
/// this via `--cf-access-client-id` + `--cf-access-client-secret`; the
/// daemon hands the same pair to every paired device. If unset the daemon
/// omits both fields from the pair-confirm response.
#[derive(Clone, Debug)]
pub struct CfAccessConfig {
    pub client_id: String,
    pub client_secret: String,
}

/// One row of the `devices.json` ledger.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DeviceRecord {
    /// Hex-encoded device public key the PWA submitted in `/pair/claim`.
    /// M8 stores it verbatim — M11+ may verify signatures with it.
    pub device_pubkey: String,
    /// Free-text label the PWA submitted (e.g. "Phone (Anna)"). Trimmed
    /// to a sane length on insert.
    pub device_name: String,
    /// 32-byte hex Bearer token. The PWA stores this encrypted under the
    /// user's passphrase via `crypto-store.js` and includes it as
    /// `Authorization: Bearer <device_token>` on every signaling request.
    pub device_token: String,
    /// ISO-8601 UTC timestamp of when the pair was confirmed.
    pub granted_at: String,
}

/// Request body for `/pair/claim`.
#[derive(Deserialize, Debug)]
pub struct ClaimReq {
    pub one_time_token: String,
    pub device_pubkey: String,
    pub device_name: String,
}

/// Response body for `/pair/claim`.
#[derive(Serialize, Debug)]
pub struct ClaimResp {
    pub ok: bool,
}

/// Request body for `/pair/confirm`.
#[derive(Deserialize, Debug)]
pub struct ConfirmReq {
    pub one_time_token: String,
    pub code: String,
}

/// Response body for `/pair/confirm`.
#[derive(Serialize, Debug)]
pub struct ConfirmResp {
    pub device_token: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cf_client_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cf_client_secret: Option<String>,
    /// Hex-encoded stable identity pubkey of the home daemon. The PWA
    /// pins this on first pair (TOFU) and refuses to talk to a daemon
    /// that hands back a different value on reconnect.
    pub peer_pubkey: String,
}

/// One pending offer waiting for the PWA to claim + confirm.
#[derive(Debug)]
struct PendingPair {
    /// 6-digit confirm code shown on the home machine.
    confirm_code: String,
    /// Filled in by `/pair/claim`. `None` until the PWA scans the QR.
    device_pubkey: Option<String>,
    /// Filled in by `/pair/claim`. `None` until the PWA scans the QR.
    device_name: Option<String>,
    /// Pending records expire after [`OFFER_TTL`] if not confirmed.
    expires_at: Instant,
}

/// What the operator's terminal (or web UI) is shown when they kick off
/// `brainwires-home pair`.
#[derive(Clone, Debug)]
pub struct PairingOffer {
    pub one_time_token: String,
    pub confirm_code: String,
}

impl PairingOffer {
    /// Build the `bwhome://pair?...` URL the PWA scans / pastes.
    ///
    /// `peer_fingerprint` is a short prefix of the daemon's `peer_pubkey`
    /// (8 hex chars) — enough for the PWA to display "pairing with home
    /// XXXXXXXX" on the confirm screen so the user can spot a swap.
    pub fn qr_url(&self, tunnel_url: &str, peer_fingerprint: &str) -> String {
        format!(
            "bwhome://pair?u={}&t={}&fp={}",
            url_encode(tunnel_url),
            url_encode(&self.one_time_token),
            url_encode(peer_fingerprint),
        )
    }
}

/// Application state for the pairing routes.
#[derive(Clone)]
pub struct PairingState {
    pending: Arc<RwLock<HashMap<String, PendingPair>>>,
    devices_path: PathBuf,
    cf_access: Option<CfAccessConfig>,
    /// Hex-encoded stable identity pubkey returned in `ConfirmResp`.
    peer_pubkey: String,
}

impl PairingState {
    /// Build a [`PairingState`] from a configured `devices.json` path,
    /// optional CF Access creds, and a stable home peer pubkey.
    pub fn new(
        devices_path: PathBuf,
        cf_access: Option<CfAccessConfig>,
        peer_pubkey: String,
    ) -> Self {
        Self {
            pending: Arc::new(RwLock::new(HashMap::new())),
            devices_path,
            cf_access,
            peer_pubkey,
        }
    }

    /// Stable identity pubkey (hex). Tests and the CLI use this to print
    /// the QR URL fingerprint.
    pub fn peer_pubkey(&self) -> &str {
        &self.peer_pubkey
    }

    /// Path to the JSON ledger this state writes to.
    pub fn devices_path(&self) -> &Path {
        &self.devices_path
    }

    /// Mint a fresh pairing offer. The token is what goes in the QR; the
    /// 6-digit code is what the user types into the PWA. Both are stored
    /// in the in-memory pending map and expire after [`OFFER_TTL`].
    pub async fn create_offer(&self) -> PairingOffer {
        let one_time_token = random_hex(TOKEN_BYTES);
        let confirm_code = random_six_digit_code();
        let pending = PendingPair {
            confirm_code: confirm_code.clone(),
            device_pubkey: None,
            device_name: None,
            expires_at: Instant::now() + OFFER_TTL,
        };
        self.pending
            .write()
            .await
            .insert(one_time_token.clone(), pending);
        PairingOffer {
            one_time_token,
            confirm_code,
        }
    }

    /// Read all stored device records. Returns an empty Vec if the file
    /// doesn't exist yet.
    pub fn read_devices(&self) -> Result<Vec<DeviceRecord>> {
        read_devices(&self.devices_path)
    }

    /// Garbage-collect expired pending offers. Called by handlers before
    /// they look the offer up so a stale token can never confirm.
    async fn gc_expired(&self) {
        let now = Instant::now();
        let mut map = self.pending.write().await;
        map.retain(|_, p| p.expires_at > now);
    }
}

/// Build the axum sub-router for pairing. Merge into the main router via
/// [`axum::Router::merge`] — the parent's CORS layer applies as usual.
pub fn router(state: PairingState) -> Router {
    Router::new()
        .route("/pair/claim", post(handle_claim))
        .route("/pair/confirm", post(handle_confirm))
        .with_state(state)
}

async fn handle_claim(
    State(s): State<PairingState>,
    Json(req): Json<ClaimReq>,
) -> impl IntoResponse {
    s.gc_expired().await;

    let mut map = s.pending.write().await;
    let Some(p) = map.get_mut(&req.one_time_token) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "unknown or expired one_time_token",
            })),
        )
            .into_response();
    };

    // Trim the operator-supplied label so a malicious PWA can't blow up
    // the JSON file with a 10 MB device_name.
    let name = req.device_name.trim();
    let trimmed_name = if name.len() > 128 {
        name.chars().take(128).collect::<String>()
    } else {
        name.to_string()
    };

    p.device_pubkey = Some(req.device_pubkey);
    p.device_name = Some(trimmed_name);
    (StatusCode::OK, Json(ClaimResp { ok: true })).into_response()
}

async fn handle_confirm(
    State(s): State<PairingState>,
    Json(req): Json<ConfirmReq>,
) -> impl IntoResponse {
    s.gc_expired().await;

    // Look up + remove in one transaction so a successful confirm is
    // single-shot (a second POST with the same token gets 404).
    let pending = {
        let mut map = s.pending.write().await;
        map.remove(&req.one_time_token)
    };

    let Some(p) = pending else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "unknown or expired one_time_token",
            })),
        )
            .into_response();
    };

    if !constant_time_eq(p.confirm_code.as_bytes(), req.code.as_bytes()) {
        // Reinsert under a fresh deadline so the user can try again? Not
        // for M8 — wrong-code is a one-shot fail to keep brute-forcing
        // expensive. Operator can run `pair` again to mint a new offer.
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": "code mismatch",
            })),
        )
            .into_response();
    }

    let Some(device_pubkey) = p.device_pubkey else {
        // Confirm came in before claim. Reject — the PWA must claim first
        // so we know which device we're authorising.
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "pair not claimed yet",
            })),
        )
            .into_response();
    };
    let device_name = p.device_name.unwrap_or_default();

    let device_token = random_hex(DEVICE_TOKEN_BYTES);
    let record = DeviceRecord {
        device_pubkey,
        device_name,
        device_token: device_token.clone(),
        granted_at: Utc::now().to_rfc3339(),
    };

    if let Err(e) = append_device(&s.devices_path, record) {
        tracing::error!(error = %e, "failed to persist device record");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "failed to persist device record",
            })),
        )
            .into_response();
    }

    let resp = ConfirmResp {
        device_token,
        cf_client_id: s.cf_access.as_ref().map(|c| c.client_id.clone()),
        cf_client_secret: s.cf_access.as_ref().map(|c| c.client_secret.clone()),
        peer_pubkey: s.peer_pubkey.clone(),
    };
    (StatusCode::OK, Json(resp)).into_response()
}

// ── on-disk persistence ───────────────────────────────────────

/// Read all records from `path`. Returns `Ok(vec![])` if missing.
pub fn read_devices(path: &Path) -> Result<Vec<DeviceRecord>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let records: Vec<DeviceRecord> =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    Ok(records)
}

/// Append a record to the JSON ledger. Atomic write (tempfile + rename),
/// `0600` permission on Unix. Creates the parent directory if needed.
pub fn append_device(path: &Path, record: DeviceRecord) -> Result<()> {
    let mut all = read_devices(path).unwrap_or_default();
    all.push(record);
    write_devices(path, &all)
}

/// Atomically write the full `records` array to `path`. Writes to
/// `<path>.tmp`, syncs, renames, and chmods to `0600` on Unix.
pub fn write_devices(path: &Path, records: &[DeviceRecord]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create_dir_all {}", parent.display()))?;
    }
    let json = serde_json::to_vec_pretty(records).context("serialize devices")?;

    let tmp = path.with_extension("json.tmp");
    {
        let mut f = fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        f.write_all(&json).context("write devices.json.tmp")?;
        f.sync_all().context("sync devices.json.tmp")?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o600);
        fs::set_permissions(&tmp, perms)
            .with_context(|| format!("chmod 0600 {}", tmp.display()))?;
    }
    fs::rename(&tmp, path)
        .with_context(|| format!("rename {} → {}", tmp.display(), path.display()))?;
    Ok(())
}

// ── identity helpers (stable peer_pubkey) ────────────────────

/// Per-daemon identity (M8 v1): a single random 32-byte hex pubkey persisted
/// at `~/.brainwires/home/identity.json`. M11+ can replace this with a real
/// Ed25519 keypair without changing the JSON shape (just add `secret_key`).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HomeIdentity {
    pub peer_pubkey: String,
}

/// Load or create the home daemon's stable identity. On first call this
/// generates a random pubkey, writes it atomically with `0600` on Unix,
/// and returns it. Subsequent calls read it back.
pub fn load_or_create_identity(path: &Path) -> Result<HomeIdentity> {
    if path.exists() {
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        if !bytes.is_empty() {
            let id: HomeIdentity = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {}", path.display()))?;
            if !id.peer_pubkey.is_empty() {
                return Ok(id);
            }
        }
    }
    let id = HomeIdentity {
        peer_pubkey: random_hex(PEER_PUBKEY_BYTES),
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create_dir_all {}", parent.display()))?;
    }
    let json = serde_json::to_vec_pretty(&id).context("serialize identity")?;
    let tmp = path.with_extension("json.tmp");
    {
        let mut f = fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        f.write_all(&json).context("write identity.json.tmp")?;
        f.sync_all().context("sync identity.json.tmp")?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o600);
        fs::set_permissions(&tmp, perms)
            .with_context(|| format!("chmod 0600 {}", tmp.display()))?;
    }
    fs::rename(&tmp, path)
        .with_context(|| format!("rename {} → {}", tmp.display(), path.display()))?;
    Ok(id)
}

/// Default brainwires-home state directory: `$HOME/.brainwires/home/`.
pub fn default_state_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".brainwires");
    p.push("home");
    Some(p)
}

// ── small helpers ─────────────────────────────────────────────

fn random_hex(n: usize) -> String {
    let mut buf = vec![0u8; n];
    rand::rng().fill_bytes(&mut buf);
    hex::encode(buf)
}

/// 6-digit numeric confirm code, padded with leading zeros. Uniform in
/// `[0, 1_000_000)`.
fn random_six_digit_code() -> String {
    let mut buf = [0u8; 4];
    rand::rng().fill_bytes(&mut buf);
    let n = u32::from_le_bytes(buf) % 1_000_000;
    format!("{:06}", n)
}

/// Constant-time bytewise compare. We avoid `subtle` here to keep the dep
/// list tight; the operation is 6 bytes and `==` would also be fine in
/// practice. Defence-in-depth — better an idiom than not.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Tiny URL-encoder for the `bwhome://` URL builder. We don't pull in
/// `urlencoding` for two fields. Encodes everything outside
/// `[A-Za-z0-9._~-]` (RFC 3986 unreserved set).
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.as_bytes() {
        let c = *b;
        let unreserved = matches!(c,
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~');
        if unreserved {
            out.push(c as char);
        } else {
            out.push_str(&format!("%{:02X}", c));
        }
    }
    out
}

// ── tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::{Method, Request};
    use serde_json::Value;
    use tempfile::TempDir;
    use tower::ServiceExt;

    fn test_state(dir: &TempDir) -> PairingState {
        let path = dir.path().join("devices.json");
        PairingState::new(
            path,
            None,
            "deadbeef".repeat(8), // 64 hex chars
        )
    }

    fn test_state_with_cf(dir: &TempDir) -> PairingState {
        let path = dir.path().join("devices.json");
        PairingState::new(
            path,
            Some(CfAccessConfig {
                client_id: "cf-id".to_string(),
                client_secret: "cf-sec".to_string(),
            }),
            "feedface".repeat(8),
        )
    }

    fn json_request(method: Method, uri: &str, body: Value) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap()
    }

    async fn body_json(resp: axum::response::Response) -> Value {
        let body = resp.into_body();
        let bytes = to_bytes(body, 1 << 20).await.expect("collect body");
        serde_json::from_slice(&bytes).expect("body is valid JSON")
    }

    #[tokio::test]
    async fn test_claim_unknown_token_returns_404() {
        let dir = TempDir::new().unwrap();
        let app = router(test_state(&dir));
        let resp = app
            .oneshot(json_request(
                Method::POST,
                "/pair/claim",
                serde_json::json!({
                    "one_time_token": "nope",
                    "device_pubkey": "abcd",
                    "device_name": "phone",
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_claim_then_confirm_happy_path() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        let app = router(state.clone());

        let offer = state.create_offer().await;

        // 1. Claim.
        let resp = app
            .clone()
            .oneshot(json_request(
                Method::POST,
                "/pair/claim",
                serde_json::json!({
                    "one_time_token": offer.one_time_token,
                    "device_pubkey": "pubkey-hex",
                    "device_name": "Anna's phone",
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["ok"].as_bool(), Some(true));

        // 2. Confirm.
        let resp = app
            .clone()
            .oneshot(json_request(
                Method::POST,
                "/pair/confirm",
                serde_json::json!({
                    "one_time_token": offer.one_time_token,
                    "code": offer.confirm_code,
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        let device_token = v["device_token"].as_str().expect("device_token");
        assert_eq!(device_token.len(), 64, "32-byte hex = 64 chars");
        assert_eq!(v["peer_pubkey"].as_str().unwrap(), state.peer_pubkey());
        assert!(v.get("cf_client_id").is_none(), "no CF config → no field");
        assert!(v.get("cf_client_secret").is_none());

        // 3. devices.json now has one record.
        let records = state.read_devices().expect("read");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].device_pubkey, "pubkey-hex");
        assert_eq!(records[0].device_name, "Anna's phone");
        assert_eq!(records[0].device_token, device_token);
        assert!(!records[0].granted_at.is_empty());
    }

    #[tokio::test]
    async fn test_confirm_includes_cf_when_configured() {
        let dir = TempDir::new().unwrap();
        let state = test_state_with_cf(&dir);
        let app = router(state.clone());

        let offer = state.create_offer().await;
        let _ = app
            .clone()
            .oneshot(json_request(
                Method::POST,
                "/pair/claim",
                serde_json::json!({
                    "one_time_token": offer.one_time_token,
                    "device_pubkey": "pk",
                    "device_name": "laptop",
                }),
            ))
            .await
            .unwrap();
        let resp = app
            .oneshot(json_request(
                Method::POST,
                "/pair/confirm",
                serde_json::json!({
                    "one_time_token": offer.one_time_token,
                    "code": offer.confirm_code,
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["cf_client_id"].as_str().unwrap(), "cf-id");
        assert_eq!(v["cf_client_secret"].as_str().unwrap(), "cf-sec");
    }

    #[tokio::test]
    async fn test_confirm_wrong_code_returns_401() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        let app = router(state.clone());

        let offer = state.create_offer().await;
        let _ = app
            .clone()
            .oneshot(json_request(
                Method::POST,
                "/pair/claim",
                serde_json::json!({
                    "one_time_token": offer.one_time_token,
                    "device_pubkey": "pk",
                    "device_name": "x",
                }),
            ))
            .await
            .unwrap();

        let resp = app
            .oneshot(json_request(
                Method::POST,
                "/pair/confirm",
                serde_json::json!({
                    "one_time_token": offer.one_time_token,
                    "code": "000000",
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // The pending offer is consumed on a wrong-code submit (single-
        // shot semantics). A second confirm must return 404.
        let dir2 = TempDir::new().unwrap();
        let state2 = test_state(&dir2);
        let app2 = router(state2.clone());
        let offer2 = state2.create_offer().await;
        let _ = app2
            .clone()
            .oneshot(json_request(
                Method::POST,
                "/pair/claim",
                serde_json::json!({
                    "one_time_token": offer2.one_time_token,
                    "device_pubkey": "pk",
                    "device_name": "x",
                }),
            ))
            .await
            .unwrap();
        let _ = app2
            .clone()
            .oneshot(json_request(
                Method::POST,
                "/pair/confirm",
                serde_json::json!({
                    "one_time_token": offer2.one_time_token,
                    "code": "999999",
                }),
            ))
            .await
            .unwrap();
        let resp = app2
            .oneshot(json_request(
                Method::POST,
                "/pair/confirm",
                serde_json::json!({
                    "one_time_token": offer2.one_time_token,
                    "code": offer2.confirm_code,
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_confirm_after_expiry_returns_404() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        let app = router(state.clone());

        let offer = state.create_offer().await;

        // Force-expire the pending entry.
        {
            let mut map = state.pending.write().await;
            let p = map.get_mut(&offer.one_time_token).unwrap();
            p.expires_at = Instant::now() - Duration::from_secs(1);
        }

        let resp = app
            .oneshot(json_request(
                Method::POST,
                "/pair/confirm",
                serde_json::json!({
                    "one_time_token": offer.one_time_token,
                    "code": offer.confirm_code,
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_confirm_without_claim_returns_400() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        let app = router(state.clone());

        let offer = state.create_offer().await;
        // Skip the claim step entirely.
        let resp = app
            .oneshot(json_request(
                Method::POST,
                "/pair/confirm",
                serde_json::json!({
                    "one_time_token": offer.one_time_token,
                    "code": offer.confirm_code,
                }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_devices_json_persisted_atomically() {
        let dir = TempDir::new().unwrap();
        let state = test_state(&dir);
        let app = router(state.clone());

        // Two consecutive happy-path pairings.
        for i in 0..2 {
            let offer = state.create_offer().await;
            let _ = app
                .clone()
                .oneshot(json_request(
                    Method::POST,
                    "/pair/claim",
                    serde_json::json!({
                        "one_time_token": offer.one_time_token,
                        "device_pubkey": format!("pubkey-{i}"),
                        "device_name": format!("device-{i}"),
                    }),
                ))
                .await
                .unwrap();
            let _ = app
                .clone()
                .oneshot(json_request(
                    Method::POST,
                    "/pair/confirm",
                    serde_json::json!({
                        "one_time_token": offer.one_time_token,
                        "code": offer.confirm_code,
                    }),
                ))
                .await
                .unwrap();
        }

        let records = state.read_devices().expect("read");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].device_pubkey, "pubkey-0");
        assert_eq!(records[1].device_pubkey, "pubkey-1");
        // Tokens are unique.
        assert_ne!(records[0].device_token, records[1].device_token);

        // Sanity: the `.tmp` shadow file should be gone after rename.
        let tmp = state.devices_path().with_extension("json.tmp");
        assert!(!tmp.exists(), "tempfile must be cleaned up after rename");
    }

    #[test]
    fn test_load_or_create_identity_round_trip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("identity.json");

        let id1 = load_or_create_identity(&path).expect("create");
        assert_eq!(id1.peer_pubkey.len(), PEER_PUBKEY_BYTES * 2);

        // Second call returns the same pubkey.
        let id2 = load_or_create_identity(&path).expect("read");
        assert_eq!(id1.peer_pubkey, id2.peer_pubkey);
    }

    #[test]
    fn test_qr_url_shape() {
        let offer = PairingOffer {
            one_time_token: "abc123".to_string(),
            confirm_code: "654321".to_string(),
        };
        let url = offer.qr_url("https://home.example.com", "deadbeef");
        assert_eq!(
            url,
            "bwhome://pair?u=https%3A%2F%2Fhome.example.com&t=abc123&fp=deadbeef"
        );
    }

    #[test]
    fn test_random_six_digit_code_padding() {
        // Run a few iterations; even rare zero-prefixed values must stay
        // 6 chars wide.
        for _ in 0..200 {
            let c = random_six_digit_code();
            assert_eq!(c.len(), 6);
            assert!(c.chars().all(|ch| ch.is_ascii_digit()));
        }
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(constant_time_eq(b"", b""));
    }
}
