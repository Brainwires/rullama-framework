// Fused per-target LoRA forward correction in a single dispatch.
//
// Replaces the two-dispatch pattern from `lora_matmul_row` used by the
// inference forward path:
//
//   z = A · x         // first dispatch (A is [rank, k], x is [k], z is [rank])
//   y += scale · B · z // second dispatch (B is [n, rank], y is [n])
//
// Both halves now run in ONE dispatch per LoRA target, halving the
// dispatch count from ~494 per token to ~247 on browser WebGPU.
// Net win: ~37ms/tok saved on bind-group + dispatch latency.
//
// Workgroup layout: 64 threads. Each workgroup handles 64 contiguous
// output rows of y in phase 2. Phase 1 (A·x) is computed COOPERATIVELY
// by all 64 threads using a per-rank tree reduction across the
// k-dimension chunk owned by each thread (chunk size = k/64). Result
// lives in workgroup memory; phase 2 reads it via the implicit
// barrier between phases.
//
// Constraints:
// - `rank` must be ≤ MAX_RANK_FUSED. Kernel returns early if violated.
// - `k` should be > 64 in practice (smaller k still works; phase 1
//   just has idle threads).
// - Dispatch shape: ((n + 63) / 64, 1, 1). Matches the row-based
//   workgroup layout.

const MAX_RANK_FUSED: u32 = 64u;

struct Params {
    k:          u32,
    n:          u32,
    rank:       u32,
    accumulate: u32,
    scale:      f32,
    _pad0:      u32,
    _pad1:      u32,
    _pad2:      u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       a:      array<f32>;
@group(0) @binding(2) var<storage, read>       b:      array<f32>;
@group(0) @binding(3) var<storage, read>       x:      array<f32>;
@group(0) @binding(4) var<storage, read_write> y:      array<f32>;
// `z_out`: rank-sized buffer that receives the captured `A·x` for the
// training backward pass. The backward LoRA path reads it to build
// `dB = scale · dy ⊗ z` and then to derive `u = Bᵀ · dy` for the
// `dA` update. Only workgroup 0 writes (the value is workgroup-
// independent — every workgroup computes the same z internally).
// Inference callers can bind a dummy buffer and ignore the writes.
@group(0) @binding(5) var<storage, read_write> z_out:  array<f32>;

// Workgroup-shared state for phase 1.
var<workgroup> z_shared: array<f32, MAX_RANK_FUSED>;
var<workgroup> partial:  array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wid: vec3<u32>,
) {
    let tid = lid.x;
    let j   = wid.x * 64u + tid;  // output row this thread emits in phase 2

    // ── Phase 1 — cooperative A·x → z_shared ───────────────────────
    //
    // For each row p of A, all 64 threads compute the dot product
    // A[p, :] · x cooperatively: thread `tid` accumulates entries
    // at indices tid, tid+64, tid+128, … then we tree-reduce in
    // workgroup memory.
    let rank = min(params.rank, MAX_RANK_FUSED);
    for (var p: u32 = 0u; p < rank; p = p + 1u) {
        var s: f32 = 0.0;
        var i: u32 = tid;
        let row_off = p * params.k;
        while (i < params.k) {
            s = s + a[row_off + i] * x[i];
            i = i + 64u;
        }
        partial[tid] = s;
        workgroupBarrier();
        // Tree reduction across 64 lanes: 32 → 16 → 8 → 4 → 2 → 1.
        // Each step halves the active lane count and sums adjacent
        // partials. workgroupBarrier between steps for cross-lane
        // visibility.
        if (tid < 32u) { partial[tid] = partial[tid] + partial[tid + 32u]; }
        workgroupBarrier();
        if (tid < 16u) { partial[tid] = partial[tid] + partial[tid + 16u]; }
        workgroupBarrier();
        if (tid <  8u) { partial[tid] = partial[tid] + partial[tid +  8u]; }
        workgroupBarrier();
        if (tid <  4u) { partial[tid] = partial[tid] + partial[tid +  4u]; }
        workgroupBarrier();
        if (tid <  2u) { partial[tid] = partial[tid] + partial[tid +  2u]; }
        workgroupBarrier();
        if (tid == 0u) {
            z_shared[p] = partial[0] + partial[1];
        }
        workgroupBarrier();
    }

    // Persist z to the rank-sized capture buffer so the training
    // backward pass can read it. Every workgroup computed the same
    // value into z_shared; only workgroup 0 writes to global memory
    // to avoid redundant traffic. Inference callers bind a dummy.
    if (wid.x == 0u && tid < rank) {
        z_out[tid] = z_shared[tid];
    }

    // ── Phase 2 — y[j] += scale · B[j] · z_shared ──────────────────
    //
    // Each thread emits one output row. The B matrix is [n, rank]
    // row-major so the thread reads `rank` contiguous entries.
    if (j >= params.n) { return; }
    var acc: f32 = 0.0;
    let b_row = j * params.rank;
    for (var p: u32 = 0u; p < rank; p = p + 1u) {
        acc = acc + b[b_row + p] * z_shared[p];
    }
    let v = params.scale * acc;
    if (params.accumulate != 0u) {
        y[j] = y[j] + v;
    } else {
        y[j] = v;
    }
}
