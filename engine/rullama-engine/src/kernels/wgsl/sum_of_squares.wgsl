// Sum-of-squares reduction: output[0] = Σ x[i]² for i in [0, n).
//
// Single-workgroup, workgroup_size=256. Each thread sums every 256th
// element into a per-thread accumulator, then a shared-memory
// tree reduction across the 256 threads. Designed for the
// LoRA gradient buffers (largest ≈ rank * ffn_inter ≈ 64 K f32s)
// where one workgroup is more than enough.
//
// `params.scale_in` multiplies the input value before squaring —
// useful for masked / weighted accumulation. Pass 1.0 to ignore.

struct Params {
    n: u32,
    scale_in: f32,
    _p0: u32,
    _p1: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       input:  array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var acc: f32 = 0.0;
    var i = tid;
    while (i < params.n) {
        let v = input[i] * params.scale_in;
        acc = acc + v * v;
        i = i + 256u;
    }
    partial[tid] = acc;
    workgroupBarrier();

    // Tree reduction across 256 lanes: 128→64→32→16→8→4→2→1.
    var stride = 128u;
    loop {
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }

    if (tid == 0u) {
        output[0] = partial[0];
    }
}
