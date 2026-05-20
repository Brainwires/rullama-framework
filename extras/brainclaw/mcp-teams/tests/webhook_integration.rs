//! Integration test for the Teams webhook: sign a Bot Framework
//! activity with a test RSA key, seed the verifier's JWKs with the
//! matching public key, POST a signed activity, and assert it is
//! forwarded to a mock gateway mpsc sink.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::routing::post;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use jsonwebtoken::{Algorithm, EncodingKey, Header};
use rsa::{RsaPrivateKey, RsaPublicKey, pkcs8::EncodePrivateKey, traits::PublicKeyParts};
use serde_json::json;

use brainwires_network::channels::ChannelEvent;
use brainwires_teams_channel::jwt::{BotFrameworkVerifier, JwkEntry};
use brainwires_teams_channel::teams::ServiceUrlStore;
use brainwires_teams_channel::webhook::{WebhookState, serve};

fn b64url(bytes: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

fn fresh_rsa_keypair() -> (String, String, String) {
    let mut rng = rand_legacy::rngs::OsRng;
    let key = RsaPrivateKey::new(&mut rng, 2048).unwrap();
    let pem = key
        .to_pkcs8_pem(rsa::pkcs8::LineEnding::LF)
        .unwrap()
        .to_string();
    let public = RsaPublicKey::from(&key);
    (
        pem,
        b64url(&public.n().to_bytes_be()),
        b64url(&public.e().to_bytes_be()),
    )
}

async fn start_webhook(
    audience: &str,
    verifier: Arc<BotFrameworkVerifier>,
    service_urls: Arc<ServiceUrlStore>,
    tx: tokio::sync::mpsc::Sender<ChannelEvent>,
) -> String {
    let state = WebhookState {
        verifier,
        service_urls,
        event_tx: tx,
    };
    let _ = audience;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let listen = addr.to_string();
    let listen_clone = listen.clone();
    tokio::spawn(async move {
        let _ = serve(state, &listen_clone).await;
    });
    for _ in 0..20 {
        if tokio::net::TcpStream::connect(&listen).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    format!("http://{}/api/messages", listen)
}

#[tokio::test]
async fn webhook_forwards_signed_activity() {
    let audience = "bot-app-id";
    let (pem, n, e) = fresh_rsa_keypair();
    let kid = "k1";

    let verifier = Arc::new(BotFrameworkVerifier::new(audience));
    verifier.seed_jwks(
        "https://local/jwks",
        vec![JwkEntry {
            kid: kid.into(),
            alg: Some("RS256".into()),
            n,
            e,
        }],
    );
    let service_urls = Arc::new(ServiceUrlStore::new());
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);

    let url = start_webhook(audience, verifier, Arc::clone(&service_urls), tx).await;

    let now = chrono::Utc::now().timestamp();
    let claims = json!({
        "aud": audience,
        "iss": "https://api.botframework.com",
        "appid": "microsoft-bot",
        "iat": now,
        "exp": now + 300,
    });
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(kid.into());
    let encoding_key = EncodingKey::from_rsa_pem(pem.as_bytes()).unwrap();
    let token = jsonwebtoken::encode(&header, &claims, &encoding_key).unwrap();

    let activity = json!({
        "type": "message",
        "id": "act-1",
        "serviceUrl": "https://smba.trafficmanager.net/amer",
        "conversation": { "id": "a:abcdef" },
        "from": { "id": "29:u", "name": "Tester" },
        "text": "hello teams",
        "timestamp": "2025-02-01T00:00:00Z",
    });

    let resp = reqwest::Client::new()
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&activity)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);

    let evt = tokio::time::timeout(Duration::from_secs(3), rx.recv())
        .await
        .expect("timeout")
        .unwrap();
    match evt {
        ChannelEvent::MessageReceived(m) => {
            assert_eq!(m.conversation.channel_id, "a:abcdef");
            assert_eq!(m.author, "Tester");
        }
        _ => panic!("unexpected event"),
    }

    assert_eq!(
        service_urls.get("a:abcdef").as_deref(),
        Some("https://smba.trafficmanager.net/amer")
    );
}

#[tokio::test]
async fn webhook_rejects_missing_auth() {
    let audience = "bot-app-id";
    let (_, n, e) = fresh_rsa_keypair();

    let verifier = Arc::new(BotFrameworkVerifier::new(audience));
    verifier.seed_jwks(
        "https://local/jwks",
        vec![JwkEntry {
            kid: "k".into(),
            alg: Some("RS256".into()),
            n,
            e,
        }],
    );
    let service_urls = Arc::new(ServiceUrlStore::new());
    let (tx, _rx) = tokio::sync::mpsc::channel(4);
    let url = start_webhook(audience, verifier, service_urls, tx).await;

    let resp = reqwest::Client::new()
        .post(&url)
        .json(&json!({ "type": "message" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn webhook_rejects_wrong_audience() {
    let audience = "expected";
    let (pem, n, e) = fresh_rsa_keypair();
    let kid = "k";
    let verifier = Arc::new(BotFrameworkVerifier::new(audience));
    verifier.seed_jwks(
        "https://local/jwks",
        vec![JwkEntry {
            kid: kid.into(),
            alg: Some("RS256".into()),
            n,
            e,
        }],
    );
    let service_urls = Arc::new(ServiceUrlStore::new());
    let (tx, _rx) = tokio::sync::mpsc::channel(4);
    let url = start_webhook(audience, verifier, service_urls, tx).await;

    let now = chrono::Utc::now().timestamp();
    let claims = json!({
        "aud": "wrong",
        "iss": "https://api.botframework.com",
        "iat": now,
        "exp": now + 60,
    });
    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(kid.into());
    let encoding_key = EncodingKey::from_rsa_pem(pem.as_bytes()).unwrap();
    let token = jsonwebtoken::encode(&header, &claims, &encoding_key).unwrap();

    let resp = reqwest::Client::new()
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&json!({ "type": "message", "conversation": { "id": "x" } }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);
}

// silence a stray axum import warning
#[allow(dead_code)]
fn _dummy() -> Router {
    Router::new().route("/", post(|| async { "" }))
}
