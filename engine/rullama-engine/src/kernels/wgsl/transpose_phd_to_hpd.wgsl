// Transpose [n_pidxes, n_heads, head_dim] → [n_heads, n_pidxes, head_dim].
//
// The current attention I/O layout (matching the matmul writeback shape) is
// pidx-major, which makes each WG's per-head K/V tile loads strided —
// consecutive K rows for one head are n_heads × head_dim = 768 f32 apart,
// so each row reads ~64 useful bytes per 3072-byte cache line fetched. On
// the Pro 555 / Metal back-end this measurably caps attention bandwidth.
// Re-laying K/V to head-major makes the per-head slice contiguous, so the
// tile-load coalesces.
//
// One thread per output element. Workgroup_size = 64.

struct Params {
    n_pidxes: u32,
    n_heads:   u32,
    head_dim:  u32,
    _pad:      u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       src:    array<f32>;
@group(0) @binding(2) var<storage, read_write> dst:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    let total: u32 = params.n_pidxes * params.n_heads * params.head_dim;
    if (i >= total) { return; }
    // Decompose i as (head, pidx, d) in the dst layout.
    let head_dim = params.head_dim;
    let n_pidxes = params.n_pidxes;
    let n_heads = params.n_heads;
    let d:     u32 = i % head_dim;
    let ph:    u32 = i / head_dim;
    let pidx: u32 = ph % n_pidxes;
    let head:  u32 = ph / n_pidxes;
    // Source index: (pidx * n_heads + head) * head_dim + d
    let src_i: u32 = (pidx * n_heads + head) * head_dim + d;
    dst[i] = src[src_i];
}
