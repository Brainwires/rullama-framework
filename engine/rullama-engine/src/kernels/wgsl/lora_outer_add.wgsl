// Rank-1 outer-product accumulator: out[i, j] += scale · a[i] · b[j].
//
// Used by LoRA backward:
//   dA[p, k] += scale · u[p] · x[k]      (a=u, b=x, out=[r, k])
//   dB[j, p] += scale · dy[j] · z[p]    (a=dy, b=z, out=[n, r])
//
// 2D dispatch (i over `outer_a`, j over `outer_b`). One thread per (i, j).

struct Params {
    outer_a:    u32,
    outer_b:    u32,
    accumulate: u32,
    _pad:       u32,
    scale:      f32,
    _pad2:      u32,
    _pad3:      u32,
    _pad4:      u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       a:      array<f32>;
@group(0) @binding(2) var<storage, read>       b:      array<f32>;
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= params.outer_a || j >= params.outer_b) { return; }
    let off = i * params.outer_b + j;
    let v = params.scale * a[i] * b[j];
    if (params.accumulate != 0u) {
        out[off] = out[off] + v;
    } else {
        out[off] = v;
    }
}
