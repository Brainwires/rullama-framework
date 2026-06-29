// Column-extract for the embed_tokens LoRA forward.
//
// A has shape [rank, vocab] row-major: A[r, v] = a[r * vocab + v].
// Computes z[r] = A[r, col] for r in 0..rank — i.e. extracts a single
// column of A indexed by `col`. This is the LoRA-side equivalent of
// `z = A @ one_hot(token_id)`: since one_hot is zero everywhere except
// position `token_id`, the matmul reduces to picking column `token_id`.
//
// Dispatch: one workgroup per `rank` (rank ≤ 64 in practice). Each
// invocation handles one row of A.

struct Params {
    rank:  u32,
    vocab: u32,
    col:   u32,    // token_id
    _pad:  u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       a:      array<f32>;
@group(0) @binding(2) var<storage, read_write> z:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= params.rank) { return; }
    z[r] = a[r * params.vocab + params.col];
}
