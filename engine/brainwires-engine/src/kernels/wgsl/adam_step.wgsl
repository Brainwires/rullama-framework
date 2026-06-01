// AdamW optimizer step — elementwise over a parameter buffer.
//
//   m_new = β₁ · m_prev + (1 - β₁) · g
//   v_new = β₂ · v_prev + (1 - β₂) · g²
//   m̂ = m_new / (1 - β₁ᵗ)
//   v̂ = v_new / (1 - β₂ᵗ)
//   p ← p - lr · ( m̂ / (√v̂ + ε) + weight_decay · p )
//
// `step` is 1-based (matches PyTorch's bias correction); pass 1 for the
// first call, 2 for the second, etc. `weight_decay` is applied
// AdamW-style — on the parameter directly, not folded into the
// gradient — so it's correctly decoupled from m/v.
//
// In-place over `param`, `m`, `v`. `grad` is read-only.

// `offset` lets the dispatcher chunk a single logical Adam step across
// multiple `dispatch_workgroups` calls so large parameter buffers
// (lm_head / embed_tokens LoRA B matrices = vocab × rank ≈ 4.2M f32s)
// don't exceed wgpu's per-dimension cap of 65_535 workgroups.

struct Params {
    n:            u32,
    step:         u32,
    offset:       u32,
    _pad1:        u32,
    lr:           f32,
    beta1:        f32,
    beta2:        f32,
    eps:          f32,
    weight_decay: f32,
    _pad2:        f32,
    _pad3:        f32,
    _pad4:        f32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       grad:   array<f32>;
@group(0) @binding(2) var<storage, read_write> param:  array<f32>;
@group(0) @binding(3) var<storage, read_write> m:      array<f32>;
@group(0) @binding(4) var<storage, read_write> v:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + params.offset;
    if (i >= params.n) { return; }
    let g = grad[i];
    let m_new = params.beta1 * m[i] + (1.0 - params.beta1) * g;
    let v_new = params.beta2 * v[i] + (1.0 - params.beta2) * g * g;
    m[i] = m_new;
    v[i] = v_new;
    let step_f = f32(params.step);
    let bc1 = 1.0 - pow(params.beta1, step_f);
    let bc2 = 1.0 - pow(params.beta2, step_f);
    let m_hat = m_new / bc1;
    let v_hat = v_new / bc2;
    let p = param[i];
    param[i] = p - params.lr * (m_hat / (sqrt(v_hat) + params.eps) + params.weight_decay * p);
}
