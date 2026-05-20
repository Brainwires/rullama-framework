//! Example: Rate-limited HTTP client with token-bucket throttling
//!
//! Demonstrates `RateLimitedClient` which wraps `reqwest::Client` with an
//! optional `RateLimiter` (token-bucket algorithm). Shows how to create
//! clients with and without rate limits, inspect available tokens, and
//! observe token consumption on each request.
//!
//! Run: cargo run -p brainwires-provider --example rate_limited_client --features native

use brainwires_provider::{RateLimitedClient, RateLimiter};

#[tokio::main]
async fn main() {
    println!("=== Rate-Limited HTTP Client Example ===\n");

    // ── 1. Create a client without rate limiting ────────────────────────
    println!("--- Client Without Rate Limiting ---");
    let unlimited = RateLimitedClient::new();
    println!(
        "  Available tokens: {:?} (None means no limiter)",
        unlimited.available_tokens()
    );
    println!();

    // ── 2. Create a client with rate limiting ───────────────────────────
    println!("--- Client With Rate Limiting (30 req/min) ---");
    let limited = RateLimitedClient::with_rate_limit(30);
    println!("  Initial tokens: {:?}", limited.available_tokens());

    // Each request consumes one token from the bucket
    println!("\n  Simulating requests (each .post()/.get() consumes a token):");
    for i in 1..=5 {
        // Building the request waits for a token, then returns a RequestBuilder.
        // We do not send the request here (no .send()), just demonstrate throttling.
        let _req = limited.post("https://httpbin.org/post").await;
        println!(
            "    Request {} -> tokens remaining: {:?}",
            i,
            limited.available_tokens()
        );
    }
    println!();

    // GET requests also consume tokens
    let _req = limited.get("https://httpbin.org/get").await;
    println!(
        "  After GET request -> tokens remaining: {:?}",
        limited.available_tokens()
    );
    println!();

    // ── 3. Using the standalone RateLimiter ─────────────────────────────
    println!("--- Standalone RateLimiter ---");
    let limiter = RateLimiter::new(60);
    println!(
        "  Max RPM: {}, Available: {}",
        limiter.max_requests_per_minute(),
        limiter.available_tokens()
    );

    // Acquire tokens directly
    for i in 1..=3 {
        limiter.acquire().await;
        println!(
            "    Acquired token {} -> remaining: {}",
            i,
            limiter.available_tokens()
        );
    }
    println!();

    // ── 4. Wrapping an existing reqwest::Client ─────────────────────────
    println!("--- From Existing reqwest::Client ---");
    let custom_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("failed to build reqwest client");

    // Wrap with rate limiting
    let wrapped = RateLimitedClient::from_client(custom_client.clone(), Some(120));
    println!("  Wrapped client tokens: {:?}", wrapped.available_tokens());

    // Wrap without rate limiting
    let unwrapped = RateLimitedClient::from_client(custom_client, None);
    println!(
        "  Unwrapped client tokens: {:?} (no limiter)",
        unwrapped.available_tokens()
    );

    // Access the inner reqwest::Client if needed
    let _inner: &reqwest::Client = wrapped.inner();
    println!("  Inner reqwest::Client accessible via .inner()");
    println!();

    // ── 5. Show how rate limiting protects against API throttling ───────
    println!("--- Rate Limiting Strategy ---");
    println!("  The RateLimiter uses a token-bucket algorithm:");
    println!("  - Tokens refill at a fixed rate (RPM / 60 per second)");
    println!("  - Each .acquire() consumes one token");
    println!("  - When tokens are exhausted, .acquire() sleeps until refill");
    println!("  - This prevents HTTP 429 (Too Many Requests) errors");
    println!();

    println!("Done! In a real application, the RateLimitedClient is used");
    println!("internally by provider clients (OpenAI, Anthropic, etc.) to");
    println!("respect API rate limits automatically.");
}
