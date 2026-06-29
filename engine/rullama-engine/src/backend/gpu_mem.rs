//! Process-global GPU buffer allocation accountant — diagnostic only.
//!
//! Tracks a running total of the bytes in every "large" `wgpu::Buffer`
//! we allocate, and logs each alloc/free above a threshold with the
//! running total. This is how we localize the iOS peak-memory ceiling:
//! instead of guessing which term tips the WebContent process over
//! jetsam, the log shows the full alloc/free trajectory with sizes and
//! the running resident total at every large event.
//!
//! On wasm32 the lines go to the browser console (`[gpumem] …`); on
//! native they're gated behind `RULLAMA_TRACE_MEM=1` so test runs stay
//! quiet by default. The counter is a best-effort accounting of OUR
//! tracked allocations — it doesn't see wgpu/driver overhead or
//! deferred WebGPU frees, but the deltas pinpoint which subsystem grows.

use std::sync::atomic::{AtomicI64, Ordering};

static TOTAL_BYTES: AtomicI64 = AtomicI64::new(0);
// Per-category running totals, so a single query returns a breakdown
// (weights vs scratch vs kv vs lora vs other) instead of one opaque
// number. Categorized by the label prefix in record_alloc/free.
static WEIGHT_BYTES: AtomicI64 = AtomicI64::new(0);
static SCRATCH_BYTES: AtomicI64 = AtomicI64::new(0);
static KV_BYTES: AtomicI64 = AtomicI64::new(0);
static LORA_BYTES: AtomicI64 = AtomicI64::new(0);
static OTHER_BYTES: AtomicI64 = AtomicI64::new(0);

fn category(label: &str) -> &'static AtomicI64 {
    if label.starts_with("weight:") {
        &WEIGHT_BYTES
    } else if label.starts_with("scratch")
        || label.starts_with("ckpt")
        || label.starts_with("layer.")
    {
        &SCRATCH_BYTES
    } else if label.starts_with("kv") {
        &KV_BYTES
    } else if label.starts_with("lora") || label.starts_with("adam") {
        &LORA_BYTES
    } else {
        &OTHER_BYTES
    }
}

/// Only log individual events at or above this size (smaller buffers
/// still count toward the total, they just don't spam a line each).
/// Native-only: wasm32 callers don't reference it (per-alloc logging is
/// suppressed on wasm to keep the WebKit console buffer from inflating).
#[cfg(not(target_arch = "wasm32"))]
const LOG_THRESHOLD: u64 = 1 << 20; // 1 MiB

/// Snapshot of the current tracked GPU buffer totals, in MiB.
/// Queryable on-demand (and folded into training beacons) so the test
/// harness reads the on-device memory trajectory instead of relying on
/// console spam. Note: this is OUR tracked `wgpu::Buffer` bytes only —
/// it excludes wgpu/Metal driver overhead, wasm linear memory, and
/// WebKit, so treat it as the controllable lower bound of the
/// WebContent footprint, not the absolute RSS.
pub fn snapshot_mib() -> (i64, i64, i64, i64, i64, i64) {
    (
        TOTAL_BYTES.load(Ordering::Relaxed) >> 20,
        WEIGHT_BYTES.load(Ordering::Relaxed) >> 20,
        SCRATCH_BYTES.load(Ordering::Relaxed) >> 20,
        KV_BYTES.load(Ordering::Relaxed) >> 20,
        LORA_BYTES.load(Ordering::Relaxed) >> 20,
        OTHER_BYTES.load(Ordering::Relaxed) >> 20,
    )
}

/// Compact one-line breakdown, e.g. `tot=1784 w=1700 s=20 kv=8 lora=3 o=53`.
pub fn breakdown_str() -> String {
    let (t, w, s, kv, l, o) = snapshot_mib();
    format!("tot={t} w={w} s={s} kv={kv} lora={l} o={o}")
}

/// Record a buffer allocation. `label` identifies the subsystem
/// (e.g. `"weight:blk.3.ffn_down"`, `"scratch.d_logits"`, `"kv.k.12"`).
///
/// Per-allocation logging is **native-only**: on wasm (iOS Safari) the
/// hundreds of `console.log`s per forward inflate the WebKit console
/// buffer and add real memory pressure to an already-tight budget, so
/// we only keep the running counter there and surface it via
/// [`mark`] / the worker's per-step `weightCacheMB` beacon. The native
/// ledger keeps the full per-buffer trail (gated by `RULLAMA_TRACE_MEM`).
pub fn record_alloc(label: &str, bytes: u64) {
    let total = TOTAL_BYTES.fetch_add(bytes as i64, Ordering::Relaxed) + bytes as i64;
    category(label).fetch_add(bytes as i64, Ordering::Relaxed);
    #[cfg(not(target_arch = "wasm32"))]
    if bytes >= LOG_THRESHOLD {
        log_line(&format!(
            "ALLOC {label} +{}MiB  total={}MiB",
            bytes >> 20,
            total >> 20
        ));
    }
    #[cfg(target_arch = "wasm32")]
    let _ = total;
}

/// Record a buffer free / destroy.
pub fn record_free(label: &str, bytes: u64) {
    let total = TOTAL_BYTES.fetch_sub(bytes as i64, Ordering::Relaxed) - bytes as i64;
    category(label).fetch_sub(bytes as i64, Ordering::Relaxed);
    #[cfg(not(target_arch = "wasm32"))]
    if bytes >= LOG_THRESHOLD {
        log_line(&format!(
            "FREE  {label} -{}MiB  total={}MiB",
            bytes >> 20,
            total >> 20
        ));
    }
    #[cfg(target_arch = "wasm32")]
    let _ = total;
}

/// Current tracked total in MiB.
pub fn total_mib() -> i64 {
    TOTAL_BYTES.load(Ordering::Relaxed) >> 20
}

/// Emit a labeled marker line with the current total — call at phase
/// boundaries (forward start, backward start, step done) to anchor the
/// trajectory.
pub fn mark(label: &str) {
    log_line(&format!("MARK  {label}  total={}MiB", total_mib()));
}

fn log_line(s: &str) {
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&format!("[gpumem] {s}")));
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        if *ON.get_or_init(|| std::env::var("RULLAMA_TRACE_MEM").is_ok()) {
            eprintln!("[gpumem] {s}");
        }
    }
}
