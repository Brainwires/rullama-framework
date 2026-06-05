//! Primitive f32 CPU ops for the Kokoro oracle. Correctness over speed.
//!
//! Activations are flat `Vec<f32>` in row-major `[rows, dim]` (row r, col c → `r*dim+c`),
//! unless noted as channel-major `[C, T]` (the conv/decoder convention, c*T + t).
#![allow(dead_code)]

/// y[t,o] = sum_i x[t,i] * w[o,i] + b[o].  w is row-major `[out, in]` (PyTorch Linear).
pub fn linear(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    w: &[f32],
    b: Option<&[f32]>,
    out_dim: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out_dim];
    for t in 0..rows {
        let xr = &x[t * in_dim..(t + 1) * in_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = b.map_or(0.0, |bb| bb[o]);
            for i in 0..in_dim {
                acc += xr[i] * wr[i];
            }
            y[t * out_dim + o] = acc;
        }
    }
    y
}

/// LayerNorm over the last dim, per row. gamma/beta length == dim.
pub fn layer_norm(
    x: &[f32],
    rows: usize,
    dim: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..dim {
            y[r * dim + c] = (row[c] - mean) * inv * gamma[c] + beta[c];
        }
    }
    y
}

/// gelu_new (tanh approximation), the HF default ALBERT activation.
pub fn gelu_new(x: &mut [f32]) {
    const K: f32 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    for v in x.iter_mut() {
        let t = *v;
        *v = 0.5 * t * (1.0 + (K * (t + 0.044715 * t * t * t)).tanh());
    }
}

pub fn leaky_relu(x: &mut [f32], slope: f32) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}

/// In-place softmax over a contiguous slice.
pub fn softmax(row: &mut [f32]) {
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for v in row.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    let inv = 1.0 / s;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// LayerNorm over the last dim with no affine (eps default 1e-5), per row.
pub fn layer_norm_plain(x: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..dim {
            y[r * dim + c] = (row[c] - mean) * inv;
        }
    }
    y
}

/// Single-direction LSTM over `x [T, in_dim]` → `[T, hidden]`. h0=c0=0.
/// PyTorch gate order in the 4H-rows: [input, forget, cell, output].
/// `w_ih [4H, in]`, `w_hh [4H, H]`, `b_ih/b_hh [4H]`. `reverse` walks t high→low.
#[allow(clippy::too_many_arguments)]
fn lstm_dir(
    x: &[f32],
    t: usize,
    in_dim: usize,
    hidden: usize,
    w_ih: &[f32],
    w_hh: &[f32],
    b_ih: &[f32],
    b_hh: &[f32],
    reverse: bool,
) -> Vec<f32> {
    let h = hidden;
    let mut out = vec![0.0f32; t * h];
    let mut h_prev = vec![0.0f32; h];
    let mut c_prev = vec![0.0f32; h];
    let mut gates = vec![0.0f32; 4 * h];
    for step in 0..t {
        let ti = if reverse { t - 1 - step } else { step };
        let xr = &x[ti * in_dim..(ti + 1) * in_dim];
        // gates = W_ih x + b_ih + W_hh h_prev + b_hh
        for r in 0..4 * h {
            let mut acc = b_ih[r] + b_hh[r];
            let ih = &w_ih[r * in_dim..(r + 1) * in_dim];
            for i in 0..in_dim {
                acc += ih[i] * xr[i];
            }
            let hh = &w_hh[r * h..(r + 1) * h];
            for i in 0..h {
                acc += hh[i] * h_prev[i];
            }
            gates[r] = acc;
        }
        for j in 0..h {
            let i_g = sigmoid(gates[j]);
            let f_g = sigmoid(gates[h + j]);
            let g_g = gates[2 * h + j].tanh();
            let o_g = sigmoid(gates[3 * h + j]);
            let c = f_g * c_prev[j] + i_g * g_g;
            let hh = o_g * c.tanh();
            c_prev[j] = c;
            h_prev[j] = hh;
            out[ti * h + j] = hh;
        }
    }
    out
}

/// Bidirectional LSTM → `[T, 2*hidden]` (forward dirs in `[..hidden]`, reverse in `[hidden..]`).
#[allow(clippy::too_many_arguments)]
pub fn bilstm(
    x: &[f32],
    t: usize,
    in_dim: usize,
    hidden: usize,
    w_ih_f: &[f32],
    w_hh_f: &[f32],
    b_ih_f: &[f32],
    b_hh_f: &[f32],
    w_ih_r: &[f32],
    w_hh_r: &[f32],
    b_ih_r: &[f32],
    b_hh_r: &[f32],
) -> Vec<f32> {
    let fwd = lstm_dir(x, t, in_dim, hidden, w_ih_f, w_hh_f, b_ih_f, b_hh_f, false);
    let rev = lstm_dir(x, t, in_dim, hidden, w_ih_r, w_hh_r, b_ih_r, b_hh_r, true);
    let mut out = vec![0.0f32; t * 2 * hidden];
    for ti in 0..t {
        out[ti * 2 * hidden..ti * 2 * hidden + hidden]
            .copy_from_slice(&fwd[ti * hidden..(ti + 1) * hidden]);
        out[ti * 2 * hidden + hidden..(ti + 1) * 2 * hidden]
            .copy_from_slice(&rev[ti * hidden..(ti + 1) * hidden]);
    }
    out
}

/// Max-abs difference between two equal-length slices (parity metric).
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "len mismatch {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}
