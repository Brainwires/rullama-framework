//! StyleTTS2 / Kokoro vocoder DSP dispatchers: 1-D convs (+ f16), transpose
//! convs, the activations they feed (leaky_relu, gelu_exact, snake, adain),
//! 2-D transpose / nearest-upsample, and the iSTFT vocoder (spec_phase +
//! istft). Their param structs live here since nothing else uses them; the
//! `cached_dispatch` / `wg_grid` helpers come from the parent module.

use bytemuck::{Pod, Zeroable};

use super::{cached_dispatch, wg_grid};
use crate::backend::WgpuCtx;
use crate::backend::pipelines::Pipelines;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Conv1dParams {
    cin: u32,
    tin: u32,
    cout: u32,
    tout: u32,
    k: u32,
    stride: u32,
    pad: u32,
    dilation: u32,
    groups: u32,
    has_bias: u32,
    _p0: u32,
    _p1: u32,
}

/// General channel-major 1-D conv. Returns `tout`. Buffers: x, w, bias, y.
/// Caller must size `y` to `cout * tout` where tout = (tin+2pad-dil*(k-1)-1)/stride+1.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    y: &wgpu::Buffer,
    cin: usize,
    tin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
    groups: usize,
) -> usize {
    let tout = (tin + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    let params = Conv1dParams {
        cin: cin as u32,
        tin: tin as u32,
        cout: cout as u32,
        tout: tout as u32,
        k: k as u32,
        stride: stride as u32,
        pad: pad as u32,
        dilation: dilation as u32,
        groups: groups as u32,
        has_bias: bias.is_some() as u32,
        _p0: 0,
        _p1: 0,
    };
    let b = bias.unwrap_or(dummy);
    let total = (cout * tout) as u32;
    cached_dispatch(
        ctx,
        enc,
        &p.conv1d,
        "conv1d",
        &[x, w, b, y],
        &params,
        wg_grid(total as usize),
    );
    tout
}

/// f16-weight variant of [`conv1d_chained`]. Identical args, except `w` is an
/// f16-packed weight buffer (2 halves per u32, `write_storage_f16` layout). x /
/// bias / y stay f32. Halves the resident weight footprint of the StyleTTS2
/// convolutions for the memory-tight f16 clone.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_f16_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    y: &wgpu::Buffer,
    cin: usize,
    tin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
    groups: usize,
) -> usize {
    let tout = (tin + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    let params = Conv1dParams {
        cin: cin as u32,
        tin: tin as u32,
        cout: cout as u32,
        tout: tout as u32,
        k: k as u32,
        stride: stride as u32,
        pad: pad as u32,
        dilation: dilation as u32,
        groups: groups as u32,
        has_bias: bias.is_some() as u32,
        _p0: 0,
        _p1: 0,
    };
    let b = bias.unwrap_or(dummy);
    let total = (cout * tout) as u32;
    cached_dispatch(
        ctx,
        enc,
        &p.conv1d_f16,
        "conv1d_f16",
        &[x, w, b, y],
        &params,
        wg_grid(total as usize),
    );
    tout
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct ConvT1dParams {
    cin: u32,
    tin: u32,
    cout: u32,
    tout: u32,
    k: u32,
    stride: u32,
    pad: u32,
    groups: u32,
    has_bias: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

/// General channel-major ConvTranspose1d (groups=1 for ISTFTNet ups, groups=C for
/// the depthwise pool). Returns tout. Buffers: x, w, bias, y.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    y: &wgpu::Buffer,
    cin: usize,
    tin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
    output_padding: usize,
    groups: usize,
) -> usize {
    let tout = (tin - 1) * stride + (k - 1) + output_padding + 1 - 2 * pad;
    let params = ConvT1dParams {
        cin: cin as u32,
        tin: tin as u32,
        cout: cout as u32,
        tout: tout as u32,
        k: k as u32,
        stride: stride as u32,
        pad: pad as u32,
        groups: groups as u32,
        has_bias: bias.is_some() as u32,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    let b = bias.unwrap_or(dummy);
    let total = (cout * tout) as u32;
    cached_dispatch(
        ctx,
        enc,
        &p.conv_transpose1d,
        "convT1d",
        &[x, w, b, y],
        &params,
        wg_grid(total as usize),
    );
    tout
}

/// f16-weight variant of [`conv_transpose1d_chained`]. `w` is an f16-packed
/// weight buffer (2 halves per u32, `write_storage_f16` layout); x/bias/y f32.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d_f16_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    bias: Option<&wgpu::Buffer>,
    dummy: &wgpu::Buffer,
    y: &wgpu::Buffer,
    cin: usize,
    tin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    pad: usize,
    output_padding: usize,
    groups: usize,
) -> usize {
    let tout = (tin - 1) * stride + (k - 1) + output_padding + 1 - 2 * pad;
    let params = ConvT1dParams {
        cin: cin as u32,
        tin: tin as u32,
        cout: cout as u32,
        tout: tout as u32,
        k: k as u32,
        stride: stride as u32,
        pad: pad as u32,
        groups: groups as u32,
        has_bias: bias.is_some() as u32,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    let b = bias.unwrap_or(dummy);
    let total = (cout * tout) as u32;
    cached_dispatch(
        ctx,
        enc,
        &p.conv_transpose1d_f16,
        "convT1d_f16",
        &[x, w, b, y],
        &params,
        wg_grid(total as usize),
    );
    tout
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct LeakyReluParams {
    n: u32,
    slope: f32,
    _p0: u32,
    _p1: u32,
}

/// Elementwise LeakyReLU in-place on `y`.
pub fn leaky_relu_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    y: &wgpu::Buffer,
    n: usize,
    slope: f32,
) {
    let params = LeakyReluParams {
        n: n as u32,
        slope,
        _p0: 0,
        _p1: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.leaky_relu,
        "leaky_relu",
        &[y],
        &params,
        wg_grid(n),
    );
}

/// Exact (erf) GELU, in-place — the StyleTTS2 diffusion denoiser activation.
pub fn gelu_exact_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    y: &wgpu::Buffer,
    n: usize,
) {
    let params = LeakyReluParams {
        n: n as u32,
        slope: 0.0,
        _p0: 0,
        _p1: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.gelu_exact,
        "gelu_exact",
        &[y],
        &params,
        wg_grid(n),
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct SnakeParams {
    c: u32,
    t: u32,
    _p0: u32,
    _p1: u32,
}

/// Snake1D activation (channel-major), per-channel alpha. Buffers: x, alpha, y.
#[allow(clippy::too_many_arguments)]
pub fn snake_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    alpha: &wgpu::Buffer,
    y: &wgpu::Buffer,
    c: usize,
    t: usize,
) {
    let params = SnakeParams {
        c: c as u32,
        t: t as u32,
        _p0: 0,
        _p1: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.snake,
        "snake",
        &[x, alpha, y],
        &params,
        wg_grid(c * t),
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct AdainParams {
    c: u32,
    t: u32,
    eps: f32,
    _p0: u32,
}

/// AdaIN1d (channel-major): per-channel InstanceNorm + (1+gamma)*·+beta. gamma/beta
/// are precomputed = chunk(fc(style)). One workgroup per channel. Buffers: x, gamma, beta, y.
#[allow(clippy::too_many_arguments)]
pub fn adain_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    gamma: &wgpu::Buffer,
    beta: &wgpu::Buffer,
    y: &wgpu::Buffer,
    c: usize,
    t: usize,
    eps: f32,
) {
    let params = AdainParams {
        c: c as u32,
        t: t as u32,
        eps,
        _p0: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.adain,
        "adain",
        &[x, gamma, beta, y],
        &params,
        (c as u32, 1, 1),
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Transpose2dParams {
    rows: u32,
    cols: u32,
    _p0: u32,
    _p1: u32,
}

/// 2-D transpose: x[rows, cols] row-major → y[cols, rows]. Buffers: x, y.
pub fn transpose2d_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) {
    let params = Transpose2dParams {
        rows: rows as u32,
        cols: cols as u32,
        _p0: 0,
        _p1: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.transpose2d,
        "transpose2d",
        &[x, y],
        &params,
        (((rows * cols) as u32).div_ceil(64), 1, 1),
    );
}

/// Nearest ×2 upsample (channel-major): x[c, t] → y[c, 2t]. Buffers: x, y.
pub fn nearest_upsample2x_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    c: usize,
    t: usize,
) {
    let params = Transpose2dParams {
        rows: c as u32,
        cols: t as u32,
        _p0: 0,
        _p1: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.nearest_upsample2x,
        "nearest2x",
        &[x, y],
        &params,
        wg_grid(c * 2 * t),
    );
}

/// ISTFTNet spectral split: post[2*nbins, t] → spec=exp(post[:nbins]), phase=sin(post[nbins:]).
/// Buffers: post, spec, phase.
pub fn spec_phase_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    post: &wgpu::Buffer,
    spec: &wgpu::Buffer,
    phase: &wgpu::Buffer,
    nbins: usize,
    t: usize,
) {
    let params = Transpose2dParams {
        rows: nbins as u32,
        cols: t as u32,
        _p0: 0,
        _p1: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.spec_phase,
        "spec_phase",
        &[post, spec, phase],
        &params,
        (((nbins * t) as u32).div_ceil(64), 1, 1),
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct IstftParams {
    nbins: u32,
    frames: u32,
    nfft: u32,
    hop: u32,
    pad: u32,
    out_len: u32,
    _p0: u32,
    _p1: u32,
}

/// Exact iSTFT (gather form). spec/phase are `[nbins, frames]` channel-major;
/// writes `y[out_len]`. Returns out_len. Buffers: spec, phase, y.
#[allow(clippy::too_many_arguments)]
pub fn istft_chained(
    ctx: &WgpuCtx,
    p: &Pipelines,
    enc: &mut wgpu::CommandEncoder,
    spec: &wgpu::Buffer,
    phase: &wgpu::Buffer,
    y: &wgpu::Buffer,
    nbins: usize,
    frames: usize,
    nfft: usize,
    hop: usize,
) -> usize {
    let pad = nfft / 2;
    let out_len = (frames - 1) * hop + nfft - 2 * pad;
    let params = IstftParams {
        nbins: nbins as u32,
        frames: frames as u32,
        nfft: nfft as u32,
        hop: hop as u32,
        pad: pad as u32,
        out_len: out_len as u32,
        _p0: 0,
        _p1: 0,
    };
    cached_dispatch(
        ctx,
        enc,
        &p.istft,
        "istft",
        &[spec, phase, y],
        &params,
        ((out_len as u32).div_ceil(64), 1, 1),
    );
    out_len
}
