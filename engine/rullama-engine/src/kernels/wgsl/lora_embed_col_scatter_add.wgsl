// Single-column accumulator for the embed_tokens LoRA backward.
//
// d_A has shape [rank, vocab] row-major: d_A[r, v] = da[r * vocab + v].
// Adds `scale * u[r]` to d_A[r, col] for r in 0..rank.
//
// This is the gradient equivalent of `d_A += scale * u ⊗ one_hot(token_id)`:
// since one_hot is zero except at position `token_id`, the outer product
// only writes to a single column of d_A.
//
// Dispatch: one workgroup per `rank`. Each invocation accumulates into
// one column-cell of d_A.

struct Params {
    rank:  u32,
    vocab: u32,
    col:   u32,    // token_id
    _pad:  u32,
    scale: f32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       u:      array<f32>;
@group(0) @binding(2) var<storage, read_write> da:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= params.rank) { return; }
    let idx = r * params.vocab + params.col;
    da[idx] = da[idx] + params.scale * u[r];
}
