// Inverse of transpose_phd_to_hpd: [n_heads, n_pidxes, head_dim]
// → [n_pidxes, n_heads, head_dim]. Used to put attention output back into
// the pidx-major layout the downstream out-projection matmul expects.

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
    let head_dim  = params.head_dim;
    let n_heads   = params.n_heads;
    let n_pidxes = params.n_pidxes;
    // Decompose i in the dst (pidx-major) layout: (pidx, head, d).
    let d:     u32 = i % head_dim;
    let ph:    u32 = i / head_dim;
    let head:  u32 = ph % n_heads;
    let pidx: u32 = ph / n_heads;
    // Source (head-major): (head * n_pidxes + pidx) * head_dim + d
    let src_i: u32 = (head * n_pidxes + pidx) * head_dim + d;
    dst[i] = src[src_i];
}
