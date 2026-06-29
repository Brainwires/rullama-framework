//! `brainwires-serve` — an OpenAI-compatible HTTP front end that hosts one GGUF
//! on the local GPU. Integration contract #3 in
//! `../../../docs/ARCHITECTURE-engine-harness.md`: native consumers (the rullama
//! app's devserver, the harness's `openai_chat` provider with a base-URL swap,
//! `curl`, etc.) reach the engine over `POST /v1/chat/completions` instead of
//! the wasm bundle or the C-ABI shim.
//!
//! The engine `Model` is `!Send` (it owns a wgpu device + single-threaded
//! GGUF fetchers), so it lives on one dedicated OS thread and requests are
//! marshalled to it over a channel. Per-request token output streams back over
//! a second channel, which the axum handler turns into SSE.
//!
//! Build + run (native only, `serve` feature):
//!   cargo run -p brainwires-engine --features serve --bin brainwires-serve -- \
//!       ~/.ollama/models/blobs/sha256-<digest> --port 11435 --model-name gemma4:e2b
//!
//! Then point any OpenAI client at `http://127.0.0.1:11435/v1`.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{mpsc, oneshot};

use brainwires_engine::api::{ChatMessage, ChatRole, Model};
use brainwires_engine::gguf::{FileFetcher, TensorFetcher};
use brainwires_engine::sampling::SamplingOptions;

/// One unit of output streamed from the model thread back to a handler.
enum TokenEvent {
    Token(String),
    Done,
    Err(String),
}

/// A generation request handed to the model thread.
struct Job {
    messages: Vec<ChatMessage>,
    sampling: SamplingOptions,
    max_tokens: u32,
    stop: Vec<String>,
    tx: mpsc::UnboundedSender<TokenEvent>,
}

#[derive(Clone)]
struct AppState {
    jobs: mpsc::UnboundedSender<Job>,
    model_name: String,
}

// ---- OpenAI request shapes (the subset we honor) ----

#[derive(Deserialize)]
struct ReqMsg {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatReq {
    #[serde(default)]
    messages: Vec<ReqMsg>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    // OpenAI `stop` is either a string or an array of strings.
    #[serde(default)]
    stop: Option<serde_json::Value>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn map_role(role: &str) -> ChatRole {
    match role {
        "system" => ChatRole::System,
        "assistant" | "model" => ChatRole::Model,
        _ => ChatRole::User,
    }
}

fn parse_stop(v: Option<serde_json::Value>) -> Vec<String> {
    match v {
        Some(serde_json::Value::String(s)) => vec![s],
        Some(serde_json::Value::Array(a)) => a
            .into_iter()
            .filter_map(|x| x.as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    }
}

fn sampling_from(req: &ChatReq) -> SamplingOptions {
    let temperature = req.temperature.unwrap_or(0.7);
    if temperature <= 0.0 {
        return SamplingOptions::greedy();
    }
    SamplingOptions {
        temperature,
        top_k: req.top_k.unwrap_or(40),
        top_p: req.top_p.unwrap_or(0.95),
        repetition_penalty: 1.0,
        seed: 0,
    }
}

/// Build a Job from a request + start it on the model thread, returning the
/// token receiver. `None` if the model thread has gone away.
fn dispatch(state: &AppState, req: &ChatReq) -> Option<mpsc::UnboundedReceiver<TokenEvent>> {
    let (tx, rx) = mpsc::unbounded_channel();
    let job = Job {
        messages: req
            .messages
            .iter()
            .map(|m| ChatMessage {
                role: map_role(&m.role),
                content: m.content.clone(),
            })
            .collect(),
        sampling: sampling_from(req),
        max_tokens: req.max_tokens.unwrap_or(256),
        stop: parse_stop(req.stop.clone()),
        tx,
    };
    state.jobs.send(job).ok().map(|_| rx)
}

async fn chat_completions(State(state): State<AppState>, Json(req): Json<ChatReq>) -> Response {
    let Some(mut rx) = dispatch(&state, &req) else {
        return error_json(503, "model unavailable");
    };
    let id = format!("chatcmpl-{}", now_secs());
    let created = now_secs();
    let model = state.model_name.clone();

    if req.stream {
        // SSE: one chunk per token, a final stop chunk, then `[DONE]`.
        let stream = futures_util::stream::unfold(
            (rx, false, id.clone(), model.clone()),
            move |(mut rx, finished, id, model)| async move {
                if finished {
                    return None;
                }
                match rx.recv().await {
                    Some(TokenEvent::Token(s)) => {
                        let chunk = json!({
                            "id": id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"content": s}, "finish_reason": null}]
                        });
                        let ev = Event::default().data(chunk.to_string());
                        Some((Ok::<_, std::convert::Infallible>(ev), (rx, false, id, model)))
                    }
                    Some(TokenEvent::Err(e)) => {
                        let chunk = json!({
                            "id": id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                            "error": {"message": e}
                        });
                        let ev = Event::default().data(chunk.to_string());
                        Some((Ok(ev), (rx, true, id, model)))
                    }
                    Some(TokenEvent::Done) | None => {
                        let chunk = json!({
                            "id": id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                        });
                        // Emit the stop chunk now; the next poll yields `[DONE]`
                        // and then ends the stream.
                        let ev = Event::default().data(chunk.to_string());
                        Some((Ok(ev), (rx, true, id, model)))
                    }
                }
            },
        );
        let done = futures_util::stream::once(async {
            Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
        });
        return Sse::new(futures_util::StreamExt::chain(stream, done)).into_response();
    }

    // Non-streaming: drain to completion, return one ChatCompletion object.
    let mut content = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            TokenEvent::Token(s) => content.push_str(&s),
            TokenEvent::Err(e) => return error_json(500, &e),
            TokenEvent::Done => break,
        }
    }
    Json(json!({
        "id": id, "object": "chat.completion", "created": created, "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]
    }))
    .into_response()
}

async fn list_models(State(state): State<AppState>) -> Response {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": now_secs(),
            "owned_by": "brainwires"
        }]
    }))
    .into_response()
}

type Response = axum::response::Response;

fn error_json(code: u16, msg: &str) -> Response {
    let status = axum::http::StatusCode::from_u16(code)
        .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(json!({"error": {"message": msg}}))).into_response()
}

/// The model thread: owns the `!Send` `Model`, serves jobs serially.
fn run_model_thread(
    path: String,
    ready: oneshot::Sender<std::result::Result<(), String>>,
    mut jobs: mpsc::UnboundedReceiver<Job>,
) {
    let fetcher: Arc<dyn TensorFetcher> = match FileFetcher::open(std::path::Path::new(&path)) {
        Ok(f) => Arc::new(f),
        Err(e) => {
            let _ = ready.send(Err(format!("open {path}: {e}")));
            return;
        }
    };
    let mut model = match pollster::block_on(Model::load_streaming(fetcher)) {
        Ok(m) => m,
        Err(e) => {
            let _ = ready.send(Err(format!("load {path}: {e}")));
            return;
        }
    };
    let _ = ready.send(Ok(()));

    while let Some(job) = jobs.blocking_recv() {
        // Each request is independent — clear the KV cache from the last one.
        model.reset_native();
        model.set_sampling_native(job.sampling);

        let prompt = model.render_chat_native(&job.messages, false);
        let prompt_ids = model.encode_tokens(&prompt);
        if prompt_ids.is_empty() {
            let _ = job.tx.send(TokenEvent::Done);
            continue;
        }

        // Prefill: feed the prompt, keep the last sampled token to seed gen.
        let mut next: u32 = 0;
        for &id in &prompt_ids {
            match pollster::block_on(model.step_native(id)) {
                Ok(n) => next = n,
                Err(e) => {
                    let _ = job.tx.send(TokenEvent::Err(e.to_string()));
                    next = u32::MAX;
                    break;
                }
            }
        }
        if next == u32::MAX {
            let _ = job.tx.send(TokenEvent::Done);
            continue;
        }

        let mut acc = String::new();
        for _ in 0..job.max_tokens {
            if model.is_eos_native(next) {
                break;
            }
            // SentencePiece word-boundary marker -> space.
            let s = model.token_str_native(next).unwrap_or_default().replace('\u{2581}', " ");
            acc.push_str(&s);
            if job.tx.send(TokenEvent::Token(s)).is_err() {
                break; // client disconnected
            }
            if job.stop.iter().any(|st| !st.is_empty() && acc.ends_with(st)) {
                break;
            }
            match pollster::block_on(model.step_native(next)) {
                Ok(n) => next = n,
                Err(e) => {
                    let _ = job.tx.send(TokenEvent::Err(e.to_string()));
                    break;
                }
            }
        }
        let _ = job.tx.send(TokenEvent::Done);
    }
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let mut path: Option<String> = None;
    let mut port: u16 = 11435;
    let mut host = "127.0.0.1".to_string();
    let mut model_name = "brainwires-engine".to_string();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--port" => port = args.next().and_then(|v| v.parse().ok()).unwrap_or(port),
            "--host" => host = args.next().unwrap_or(host),
            "--model-name" => model_name = args.next().unwrap_or(model_name),
            "-h" | "--help" => {
                eprintln!(
                    "usage: brainwires-serve <gguf-path> [--port N] [--host ADDR] [--model-name NAME]"
                );
                return;
            }
            other if path.is_none() => path = Some(other.to_string()),
            _ => {}
        }
    }
    let Some(path) = path else {
        eprintln!("error: missing <gguf-path>");
        eprintln!("usage: brainwires-serve <gguf-path> [--port N] [--host ADDR] [--model-name NAME]");
        std::process::exit(2);
    };

    let (jobs_tx, jobs_rx) = mpsc::unbounded_channel::<Job>();
    let (ready_tx, ready_rx) = oneshot::channel();
    eprintln!("loading {path} ...");
    std::thread::spawn(move || run_model_thread(path, ready_tx, jobs_rx));

    match ready_rx.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("error: model thread died during load");
            std::process::exit(1);
        }
    }

    let state = AppState { jobs: jobs_tx, model_name: model_name.clone() };
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .with_state(state);

    let addr = format!("{host}:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("brainwires-serve: {model_name} on http://{addr}/v1");
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("server error: {e}");
        std::process::exit(1);
    }
}
