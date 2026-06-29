// 2D NeoX RoPE for the Gemma 4 vision tower.
//
// Per Ollama (model_vision.go lines 191–210), each head's first half of head_dim
// is rotated by NeoX RoPE using the patch's X position; the second half is
// rotated by NeoX RoPE using the patch's Y position. Single dispatch handles
// both halves and all heads in parallel.
//
// Layout: x is [head_dim, n_heads, n_patches] flat, fastest-varying axis = head_dim.
// We expect head_dim % 2 == 0 (true for Gemma 4 vision: head_dim = 768/12 = 64).
//
// The per-head element at (h, p, d) is at index `(p * n_heads + h) * head_dim + d`.
//
// rope_theta = 100.0 (NOT the text model's 10000).

struct Params {
    head_dim: u32,
    n_heads:  u32,
    n_patches: u32,
    base:     f32,
}

@group(0) @binding(0) var<uniform>             params:  Params;
@group(0) @binding(1) var<storage, read_write> x:       array<f32>;
@group(0) @binding(2) var<storage, read>       pos_x:   array<u32>;
@group(0) @binding(3) var<storage, read>       pos_y:   array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // One thread per (patch, head, half-pair). The 2D RoPE rotates pairs (i, i+halfDim/2)
    // within each half of head_dim. NeoX rotation: pair (low, high) where low = first
    // half of the chosen half-axis and high = second half. So in a 64-d head split into
    // two 32-d halves, the X half is dims [0..32) and rotates pairs (0,16),(1,17),...,(15,31);
    // the Y half is dims [32..64) and rotates pairs (32,48),(33,49),...,(47,63).
    let half_dim: u32 = params.head_dim / 2u;          // 32 for ViT
    let quarter:  u32 = half_dim / 2u;                 // 16 for ViT (each NeoX block-half)
    let total: u32 = params.n_patches * params.n_heads * half_dim;
    let i: u32 = gid.x;
    if (i >= total) { return; }

    // i decomposes as: (patch * n_heads + head) * half_dim + slot
    let slot: u32 = i % half_dim;
    let ph: u32 = i / half_dim;
    let head: u32 = ph % params.n_heads;
    let pidx: u32 = ph / params.n_heads;

    let row_base: u32 = (pidx * params.n_heads + head) * params.head_dim;

    // Are we in the X half (first 32 dims) or the Y half (last 32 dims)?
    // We dispatch one thread per slot in 0..half_dim, but we need to do two rotations
    // (one per half) per dispatch position. Easier: each thread handles both halves
    // for its slot.
    if (slot >= quarter) { return; }

    let pos_xv: f32 = f32(pos_x[pidx]);
    let pos_yv: f32 = f32(pos_y[pidx]);

    // theta_i = base ^ (-2i / half_dim), i in 0..quarter
    let exponent: f32 = -2.0 * f32(slot) / f32(half_dim);
    let theta: f32 = pow(params.base, exponent);
    let cos_x: f32 = cos(pos_xv * theta);
    let sin_x: f32 = sin(pos_xv * theta);
    let cos_y: f32 = cos(pos_yv * theta);
    let sin_y: f32 = sin(pos_yv * theta);

    // X half: dims [0..half_dim). NeoX pairs (slot, slot + quarter).
    let lo_x: u32 = row_base + slot;
    let hi_x: u32 = row_base + slot + quarter;
    let a_x: f32 = x[lo_x];
    let b_x: f32 = x[hi_x];
    x[lo_x] = a_x * cos_x - b_x * sin_x;
    x[hi_x] = a_x * sin_x + b_x * cos_x;

    // Y half: dims [half_dim..head_dim). NeoX pairs (half_dim + slot, half_dim + slot + quarter).
    let lo_y: u32 = row_base + half_dim + slot;
    let hi_y: u32 = row_base + half_dim + slot + quarter;
    let a_y: f32 = x[lo_y];
    let b_y: f32 = x[hi_y];
    x[lo_y] = a_y * cos_y - b_y * sin_y;
    x[hi_y] = a_y * sin_y + b_y * cos_y;
}
