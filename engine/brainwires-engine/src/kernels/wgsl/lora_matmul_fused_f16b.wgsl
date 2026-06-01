// Fused per-target LoRA forward correction with B stored as packed f16.
//
// Variant of `lora_matmul_fused.wgsl` for the `lm_head` LoRA only.
// B is `[vocab=262144, rank]` ≈ 16 MB at f32; packing two f16 elements
// per u32 halves bandwidth on the phase-2 B·z read, which is the
// dominant cost of the lm_head LoRA correction at vocab scale. Used
// only when the inference adapter has been built with `b_is_f16=true`
// for the lm_head slot (see `lora.rs`).
//
// Storage layout: `b_packed` is `array<u32>` with length `(n * rank) / 2`.
// Element B[j, p] lives in word `(j * rank + p) / 2`; even `p` is the
// `.x` lane, odd `p` is the `.y` lane.
//
// Constraint: `rank` must be even. Garlic-style recipes use rank=16, so
// this is satisfied in practice. Phase 2 loops `p` in strides of 2 to
// process one u32 word per step.
//
// Phase 1 (A·x) is byte-identical to the f32-B kernel — A is still f32.

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

@group(0) @binding(0) var<uniform>             params:   Params;
@group(0) @binding(1) var<storage, read>       a:        array<f32>;
// B as packed f16 pairs (u32 holds two consecutive elements along the
// last/contiguous dim, i.e. along `rank`). Decode with `unpack2x16float`.
@group(0) @binding(2) var<storage, read>       b_packed: array<u32>;
@group(0) @binding(3) var<storage, read>       x:        array<f32>;
@group(0) @binding(4) var<storage, read_write> y:        array<f32>;
@group(0) @binding(5) var<storage, read_write> z_out:    array<f32>;

var<workgroup> z_shared: array<f32, MAX_RANK_FUSED>;
var<workgroup> partial:  array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wid: vec3<u32>,
) {
    let tid = lid.x;
    let j   = wid.x * 64u + tid;

    // ── Phase 1 — cooperative A·x → z_shared (same as f32-B variant) ──
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

    if (wid.x == 0u && tid < rank) {
        z_out[tid] = z_shared[tid];
    }

    // ── Phase 2 — y[j] += scale · unpack(B[j])·z_shared ───────────────
    //
    // Loop p in strides of 2: each u32 word holds two consecutive
    // f16 elements (p and p+1). `b_row_base = j * rank` lands on an
    // even word boundary because rank is required to be even — the
    // single u32 fetch yields both lanes via `unpack2x16float`.
    if (j >= params.n) { return; }
    var acc: f32 = 0.0;
    let b_row_base = j * params.rank;
    for (var p: u32 = 0u; p < rank; p = p + 2u) {
        let word_idx = (b_row_base + p) / 2u;
        let pair     = unpack2x16float(b_packed[word_idx]);
        acc = acc + pair.x * z_shared[p];
        acc = acc + pair.y * z_shared[p + 1u];
    }
    let v = params.scale * acc;
    if (params.accumulate != 0u) {
        y[j] = y[j] + v;
    } else {
        y[j] = v;
    }
}
