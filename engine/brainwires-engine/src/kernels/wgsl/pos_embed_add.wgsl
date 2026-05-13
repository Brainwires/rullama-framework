// Add 2D position embeddings to per-patch hidden states.
//
// hidden:    f32 [n_patches, hidden_size]    (patch-major, in-place)
// pos_embd:  f32 [hidden_size, pos_size, 2]  (GGUF dim[0]=hidden fastest)
//                stacked as [X-table | Y-table] along the slow axis.
// pos_x/y:   u32 [n_patches]
//
// hidden[p, d] += pos_embd_X[posX[p]][d] + pos_embd_Y[posY[p]][d]
//
// Mirrors model_vision.go::Forward lines 306–324.

struct Params {
    n_patches:   u32,
    hidden_size: u32,
    pos_size:    u32,    // 10240 in gemma4:e2b
    _pad:        u32,
}

@group(0) @binding(0) var<uniform>             params:   Params;
@group(0) @binding(1) var<storage, read_write> hidden:   array<f32>;
@group(0) @binding(2) var<storage, read>       pos_embd: array<f32>;
@group(0) @binding(3) var<storage, read>       pos_x:    array<u32>;
@group(0) @binding(4) var<storage, read>       pos_y:    array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.n_patches * params.hidden_size;
    let idx: u32 = gid.x;
    if (idx >= total) { return; }

    let p: u32 = idx / params.hidden_size;
    let d: u32 = idx - p * params.hidden_size;

    let px: u32 = pos_x[p];
    let py: u32 = pos_y[p];

    let tbl_x: f32 = pos_embd[px * params.hidden_size + d];
    let y_base: u32 = params.pos_size * params.hidden_size;
    let tbl_y: f32 = pos_embd[y_base + py * params.hidden_size + d];

    hidden[idx] = hidden[idx] + tbl_x + tbl_y;
}
