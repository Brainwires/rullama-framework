// Cross-entropy forward + backward over a single logit vector.
//
// One workgroup processes the whole vocab via two reductions (max,
// sum-exp) and one elementwise pass:
//
//   softmax[i] = exp(logits[i] - max) / sum_exp(logits - max)
//   d_logits[i] = softmax[i] - 1[i == target]
//   loss = -log(softmax[target])
//
// When `target == u32::MAX` (the masking sentinel used by the dataset
// loader's next-token shift) or `target >= vocab_size`, both the
// gradient and the loss are zero — the kernel is safe to call on
// masked positions without a host-side branch.
//
// One workgroup, 256 threads. Each thread strides over the vocab so a
// 262 144-entry Gemma 4 logit vector dispatches in three sequential
// sweeps without launching multiple WGs (which would need a second
// reduction kernel).

struct Params {
    vocab_size: u32,
    target_id:  u32,
    _pad0:      u32,
    _pad1:      u32,
}

const WG_SIZE: u32 = 256u;
const TARGET_MASK: u32 = 0xFFFFFFFFu;
// Most-negative finite f32; safe init for max-reduce.
const NEG_INF: f32 = -3.4028235e38;

@group(0) @binding(0) var<uniform>             params:   Params;
@group(0) @binding(1) var<storage, read>       logits:   array<f32>;
@group(0) @binding(2) var<storage, read_write> d_logits: array<f32>;
@group(0) @binding(3) var<storage, read_write> loss_out: array<f32>;

var<workgroup> wg_scratch: array<f32, WG_SIZE>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.vocab_size;
    let tgt = params.target_id;
    let masked = (tgt == TARGET_MASK) || (tgt >= n);

    // ---- pass 1: max(logits) ----
    var local_max: f32 = NEG_INF;
    var i: u32 = tid;
    loop {
        if (i >= n) { break; }
        local_max = max(local_max, logits[i]);
        i = i + WG_SIZE;
    }
    wg_scratch[tid] = local_max;
    workgroupBarrier();
    var stride: u32 = WG_SIZE >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            wg_scratch[tid] = max(wg_scratch[tid], wg_scratch[tid + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let logit_max = wg_scratch[0];
    workgroupBarrier();

    // ---- pass 2: sum exp(logits - max) ----
    var local_sum: f32 = 0.0;
    i = tid;
    loop {
        if (i >= n) { break; }
        local_sum = local_sum + exp(logits[i] - logit_max);
        i = i + WG_SIZE;
    }
    wg_scratch[tid] = local_sum;
    workgroupBarrier();
    stride = WG_SIZE >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            wg_scratch[tid] = wg_scratch[tid] + wg_scratch[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let sum_exp = wg_scratch[0];
    workgroupBarrier();
    let inv_sum = 1.0 / sum_exp;

    // ---- pass 3: write softmax - one_hot(target) into d_logits ----
    i = tid;
    loop {
        if (i >= n) { break; }
        let soft = exp(logits[i] - logit_max) * inv_sum;
        if (masked) {
            d_logits[i] = 0.0;
        } else if (i == tgt) {
            d_logits[i] = soft - 1.0;
        } else {
            d_logits[i] = soft;
        }
        i = i + WG_SIZE;
    }

    // ---- loss ----
    if (tid == 0u) {
        if (masked) {
            loss_out[0] = 0.0;
        } else {
            // -log softmax[tgt] = -(logits[tgt] - max) + log(sum_exp)
            loss_out[0] = -(logits[tgt] - logit_max) + log(sum_exp);
        }
    }
}
