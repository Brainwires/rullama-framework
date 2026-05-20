//! Integration test: sign a JWT with a test RSA key, serve a mock JWKs
//! endpoint with the matching public key, POST a signed Chat event to
//! the adapter webhook, and assert that the event is forwarded to a
//! mock gateway via an mpsc channel.
//!
//! No network: the JWKs server is a second Axum instance bound to a
//! local ephemeral port.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::{Json, Router, extract::State, routing::get};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use jsonwebtoken::{Algorithm, EncodingKey, Header};
use parking_lot::Mutex;
use rsa::{RsaPrivateKey, RsaPublicKey, pkcs8::EncodePrivateKey, traits::PublicKeyParts};
use serde_json::json;

use brainwires_google_chat_channel::webhook::{WebhookState, serve};
use brainwires_network::channels::ChannelEvent;

/// Shared state for the mock JWKs server — holds a single public key.
#[derive(Clone)]
struct JwksState {
    jwks: Arc<Mutex<serde_json::Value>>,
}

async fn jwks_handler(State(state): State<JwksState>) -> Json<serde_json::Value> {
    Json(state.jwks.lock().clone())
}

async fn start_jwks_server(jwks: serde_json::Value) -> String {
    let state = JwksState {
        jwks: Arc::new(Mutex::new(jwks)),
    };
    let app = Router::new()
        .route("/jwks", get(jwks_handler))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{}/jwks", addr)
}

async fn start_webhook_server(
    audience: &str,
    jwks_url: &str,
    event_tx: tokio::sync::mpsc::Sender<ChannelEvent>,
) -> String {
    let state = WebhookState::new(audience.to_string(), event_tx).with_jwks_url(jwks_url);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let listen = addr.to_string();
    let listen_clone = listen.clone();
    tokio::spawn(async move {
        let _ = serve(state, &listen_clone).await;
    });
    // Wait for the listener to come up.
    for _ in 0..20 {
        if tokio::net::TcpStream::connect(&listen).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    format!("http://{}/events", listen)
}

fn b64url(bytes: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

/// Generate a fresh RSA key, return (encoding_key_pem, jwk_n_b64,
/// jwk_e_b64). Uses a small modulus to keep the test fast.
fn fresh_rsa_keypair() -> (String, String, String) {
    let mut rng = rand_legacy::rngs::OsRng;
    let key = RsaPrivateKey::new(&mut rng, 2048).expect("generate rsa");
    let pem = key.to_pkcs8_pem(rsa::pkcs8::LineEnding::LF).unwrap();
    let public = RsaPublicKey::from(&key);
    let n = b64url(&public.n().to_bytes_be());
    let e = b64url(&public.e().to_bytes_be());
    (pem.to_string(), n, e)
}

#[tokio::test]
async fn webhook_forwards_signed_message_to_gateway() {
    let (pem, n, e) = fresh_rsa_keypair();
    let kid = "test-kid-1";
    let jwks = json!({
        "keys": [{
            "kid": kid,
            "alg": "RS256",
            "kty": "RSA",
            "use": "sig",
            "n": n,
            "e": e,
        }]
    });
    let jwks_url = start_jwks_server(jwks).await;

    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    let audience = "https://example.test/brainclaw";
    let webhook_url = start_webhook_server(audience, &jwks_url, tx).await;

    // Build a valid Google-style ID token.
    let now = chrono::Utc::now().timestamp();
    let claims = json!({
        "iss": "https://accounts.google.com",
        "aud": audience,
        "sub": "1234567890",
        "email": "bot@project.iam.gserviceaccount.com",
        "iat": now,
        "exp": now + 300,
    });
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(kid.to_string());
    let encoding_key = EncodingKey::from_rsa_pem(pem.as_bytes()).unwrap();
    let token = jsonwebtoken::encode(&header, &claims, &encoding_key).unwrap();

    let event_body = json!({
        "type": "MESSAGE",
        "message": {
            "name": "spaces/AAA/messages/BBB",
            "sender": { "name": "users/9", "displayName": "Test User" },
            "space": { "name": "spaces/AAA" },
            "text": "hi brainclaw",
            "argumentText": "hi brainclaw",
            "createTime": "2025-01-01T00:00:00Z",
        }
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(&webhook_url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&event_body)
        .send()
        .await
        .expect("post webhook");
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let received = tokio::time::timeout(Duration::from_secs(3), rx.recv())
        .await
        .expect("timeout waiting for event")
        .expect("sender dropped");
    match received {
        ChannelEvent::MessageReceived(m) => {
            assert_eq!(m.conversation.channel_id, "AAA");
            assert_eq!(m.author, "Test User");
        }
        other => panic!("unexpected event: {other:?}"),
    }
}

#[tokio::test]
async fn webhook_rejects_missing_authorization() {
    let (_, n, e) = fresh_rsa_keypair();
    let jwks = json!({
        "keys": [{ "kid": "k", "alg": "RS256", "kty": "RSA", "n": n, "e": e }]
    });
    let jwks_url = start_jwks_server(jwks).await;

    let (tx, _rx) = tokio::sync::mpsc::channel(4);
    let webhook_url = start_webhook_server("any", &jwks_url, tx).await;

    let resp = reqwest::Client::new()
        .post(&webhook_url)
        .json(&json!({ "type": "MESSAGE" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn webhook_rejects_wrong_audience() {
    let (pem, n, e) = fresh_rsa_keypair();
    let kid = "k";
    let jwks = json!({
        "keys": [{
            "kid": kid, "alg": "RS256", "kty": "RSA", "n": n, "e": e,
        }]
    });
    let jwks_url = start_jwks_server(jwks).await;
    let (tx, _rx) = tokio::sync::mpsc::channel(4);
    let webhook_url = start_webhook_server("expected-aud", &jwks_url, tx).await;

    let now = chrono::Utc::now().timestamp();
    let claims = json!({
        "iss": "https://accounts.google.com",
        "aud": "wrong-aud",
        "sub": "1",
        "iat": now,
        "exp": now + 60,
    });
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(kid.to_string());
    let encoding_key = EncodingKey::from_rsa_pem(pem.as_bytes()).unwrap();
    let token = jsonwebtoken::encode(&header, &claims, &encoding_key).unwrap();

    let resp = reqwest::Client::new()
        .post(&webhook_url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&json!({ "type": "MESSAGE" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);
}
