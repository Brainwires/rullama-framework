//! GGML quantization block layouts and CPU dequantizers.
//!
//! v1 supports Q4_K and Q6_K (the mix in Q4_K_M Gemma 4 GGUFs), plus F16 and F32. Other
//! quants surface as an error. Source of truth: `ggml-quants.c` in llama.cpp —
//! specifically `dequantize_row_q4_K` and `dequantize_row_q6_K`.

use bytemuck::{Pod, Zeroable};
use half::f16;

use super::dtype::GgmlDtype;
use crate::error::{Result, RullamaError};

/// Number of elements in a single Q4_K / Q6_K super-block.
pub const QK_K: usize = 256;

// ---------- Q4_K ----------
//
// Block layout (144 bytes total, exactly QK_K = 256 elements):
//   d        : f16            (super-block scale)
//   dmin     : f16            (super-block min)
//   scales   : 12 bytes       (8 × 6-bit scale + 8 × 6-bit min, packed)
//   qs       : 128 bytes      (256 × 4-bit quants, two per byte)

pub const Q4_K_BLOCK_BYTES: usize = 144;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlockQ4K {
    d_bits: u16,
    dmin_bits: u16,
    scales: [u8; 12],
    qs: [u8; 128],
}

/// Decode a 6-bit `(scale, min)` pair from the 12-byte packed `scales` field.
///
/// Mirrors `get_scale_min_k4` in ggml-quants.c.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize a Q4_K-encoded byte stream into `out` (length = number of blocks × 256).
pub fn dequant_q4_k(src: &[u8], out: &mut [f32]) -> Result<()> {
    if !src.len().is_multiple_of(Q4_K_BLOCK_BYTES) {
        return Err(RullamaError::Gguf(format!(
            "Q4_K source not multiple of {Q4_K_BLOCK_BYTES} bytes (got {})",
            src.len()
        )));
    }
    let nb = src.len() / Q4_K_BLOCK_BYTES;
    if out.len() != nb * QK_K {
        return Err(RullamaError::Gguf(format!(
            "Q4_K dest expected {} elements, got {}",
            nb * QK_K,
            out.len()
        )));
    }

    let blocks: &[BlockQ4K] = bytemuck::cast_slice(src);
    for (bi, blk) in blocks.iter().enumerate() {
        let d = f16::from_bits(blk.d_bits).to_f32();
        let dmin = f16::from_bits(blk.dmin_bits).to_f32();

        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for j in 0..8 {
            let (s, m) = get_scale_min_k4(j, &blk.scales);
            scales[j] = s;
            mins[j] = m;
        }

        let dst = &mut out[bi * QK_K..(bi + 1) * QK_K];
        let mut is = 0usize;
        let mut j = 0usize;
        while j < QK_K {
            // 64 elements per iteration: 32 from low nibbles, 32 from high nibbles
            let q = &blk.qs[j / 2..j / 2 + 32];
            let s_lo = scales[is] as f32;
            let m_lo = mins[is] as f32;
            let s_hi = scales[is + 1] as f32;
            let m_hi = mins[is + 1] as f32;
            for l in 0..32 {
                dst[j + l] = d * s_lo * (q[l] & 0xF) as f32 - dmin * m_lo;
                dst[j + l + 32] = d * s_hi * (q[l] >> 4) as f32 - dmin * m_hi;
            }
            is += 2;
            j += 64;
        }
    }
    Ok(())
}

// ---------- Q6_K ----------
//
// Block layout (210 bytes total, exactly QK_K = 256 elements):
//   ql       : 128 bytes      (256 × low 4 bits)
//   qh       : 64 bytes       (256 × upper 2 bits, packed 4 per byte)
//   scales   : 16 i8          (16 × 8-bit scales, one per 16 elements)
//   d        : f16            (super-block scale)

pub const Q6_K_BLOCK_BYTES: usize = 210;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlockQ6K {
    ql: [u8; 128],
    qh: [u8; 64],
    scales: [i8; 16],
    d_bits: u16,
}

/// Dequantize a Q6_K-encoded byte stream into `out` (length = number of blocks × 256).
pub fn dequant_q6_k(src: &[u8], out: &mut [f32]) -> Result<()> {
    if !src.len().is_multiple_of(Q6_K_BLOCK_BYTES) {
        return Err(RullamaError::Gguf(format!(
            "Q6_K source not multiple of {Q6_K_BLOCK_BYTES} bytes (got {})",
            src.len()
        )));
    }
    let nb = src.len() / Q6_K_BLOCK_BYTES;
    if out.len() != nb * QK_K {
        return Err(RullamaError::Gguf(format!(
            "Q6_K dest expected {} elements, got {}",
            nb * QK_K,
            out.len()
        )));
    }

    let blocks: &[BlockQ6K] = bytemuck::cast_slice(src);
    for (bi, blk) in blocks.iter().enumerate() {
        let d = f16::from_bits(blk.d_bits).to_f32();
        let dst = &mut out[bi * QK_K..(bi + 1) * QK_K];

        // QK_K = 256, processed in 2 passes of 128 elements each.
        // Each pass uses 64 bytes of ql, 32 bytes of qh, 8 scales.
        for n_pass in 0..2 {
            let ql = &blk.ql[n_pass * 64..(n_pass + 1) * 64];
            let qh = &blk.qh[n_pass * 32..(n_pass + 1) * 32];
            let sc = &blk.scales[n_pass * 8..(n_pass + 1) * 8];
            let base = n_pass * 128;

            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) as i32 | ((qh[l] & 3) as i32) << 4) - 32;
                let q2 = ((ql[l + 32] & 0xF) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
                let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
                let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;

                dst[base + l] = d * sc[is] as f32 * q1 as f32;
                dst[base + l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                dst[base + l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                dst[base + l + 96] = d * sc[is + 6] as f32 * q4 as f32;
            }
        }
    }
    Ok(())
}

// ---------- F16 / F32 ----------

/// Convert a BF16 byte stream (little-endian, high 16 bits of an IEEE-754 f32) to f32.
pub fn bf16_to_f32(src: &[u8], out: &mut [f32]) -> Result<()> {
    if !src.len().is_multiple_of(2) {
        return Err(RullamaError::Gguf(format!(
            "BF16 source byte length {} is odd",
            src.len()
        )));
    }
    if out.len() * 2 != src.len() {
        return Err(RullamaError::Gguf(format!(
            "BF16 dest expected {} elements, got {}",
            src.len() / 2,
            out.len()
        )));
    }
    for (i, chunk) in src.chunks_exact(2).enumerate() {
        let bits = u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16;
        out[i] = f32::from_bits(bits);
    }
    Ok(())
}

/// Convert an F16 byte stream (little-endian half-precision) to f32.
pub fn f16_to_f32(src: &[u8], out: &mut [f32]) -> Result<()> {
    if !src.len().is_multiple_of(2) {
        return Err(RullamaError::Gguf(format!(
            "F16 source byte length {} is odd",
            src.len()
        )));
    }
    if out.len() * 2 != src.len() {
        return Err(RullamaError::Gguf(format!(
            "F16 dest expected {} elements, got {}",
            src.len() / 2,
            out.len()
        )));
    }
    for (i, chunk) in src.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out[i] = f16::from_bits(bits).to_f32();
    }
    Ok(())
}

/// Copy F32 little-endian bytes into a f32 vector.
pub fn f32_to_f32(src: &[u8], out: &mut [f32]) -> Result<()> {
    if !src.len().is_multiple_of(4) {
        return Err(RullamaError::Gguf(format!(
            "F32 source byte length {} not /4",
            src.len()
        )));
    }
    if out.len() * 4 != src.len() {
        return Err(RullamaError::Gguf(format!(
            "F32 dest expected {} elements, got {}",
            src.len() / 4,
            out.len()
        )));
    }
    for (i, chunk) in src.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(())
}

/// Dispatch dequant by dtype. Returns an error for unsupported types in v1.
pub fn dequant_into_f32(dtype: GgmlDtype, src: &[u8], out: &mut [f32]) -> Result<()> {
    match dtype {
        GgmlDtype::F32 => f32_to_f32(src, out),
        GgmlDtype::F16 => f16_to_f32(src, out),
        GgmlDtype::BF16 => bf16_to_f32(src, out),
        GgmlDtype::Q4_K => dequant_q4_k(src, out),
        GgmlDtype::Q6_K => dequant_q6_k(src, out),
        other => Err(RullamaError::Gguf(format!(
            "dtype {other:?} is not in v1 dequant scope (only F32, F16, BF16, Q4_K, Q6_K)"
        ))),
    }
}

/// F16 little-endian bytes → raw f16 bit patterns (passthrough, no precision
/// change). Used by the f16-resident StyleTTS2 path to keep big weights f16
/// in host memory instead of expanding them to f32.
pub fn f16_to_f16_bits(src: &[u8], out: &mut [u16]) -> Result<()> {
    if !src.len().is_multiple_of(2) {
        return Err(RullamaError::Gguf(format!("F16 source byte length {} is odd", src.len())));
    }
    if out.len() * 2 != src.len() {
        return Err(RullamaError::Gguf(format!(
            "F16 dest expected {} elements, got {}",
            src.len() / 2,
            out.len()
        )));
    }
    for (i, chunk) in src.chunks_exact(2).enumerate() {
        out[i] = u16::from_le_bytes([chunk[0], chunk[1]]);
    }
    Ok(())
}

/// Dequantize any supported dtype into raw f16 bit patterns (little-endian
/// half). F16 passes through losslessly; everything else is dequantized to f32
/// and downcast to f16. `out[i]` holds the u16 bit pattern — upload as 2 LE
/// bytes per element (the same layout `write_storage_f16` produces).
pub fn dequant_into_f16(dtype: GgmlDtype, src: &[u8], out: &mut [u16]) -> Result<()> {
    match dtype {
        GgmlDtype::F16 => f16_to_f16_bits(src, out),
        _ => {
            let mut tmp = vec![0f32; out.len()];
            dequant_into_f32(dtype, src, &mut tmp)?;
            for (o, &v) in out.iter_mut().zip(tmp.iter()) {
                *o = f16::from_f32(v).to_bits();
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesize a Q4_K block where d=1.0, dmin=0.0, scales[j]=1, mins[j]=0, and
    /// all 4-bit quants = 0. Result: every element is 0.
    fn synth_q4_k_zero() -> Vec<u8> {
        let mut buf = vec![0u8; Q4_K_BLOCK_BYTES];
        // d=1.0 in f16 = 0x3C00
        buf[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        buf[2..4].copy_from_slice(&0x0000u16.to_le_bytes()); // dmin=0
        // Set scales[0..7] = 1 in the 6-bit packing.
        // get_scale_min_k4(j<4): d=q[j]&63, m=q[j+4]&63
        // get_scale_min_k4(j>=4): d=(q[j+4]&0xF)|((q[j-4]>>6)<<4), m=(q[j+4]>>4)|((q[j]>>6)<<4)
        //
        // For scales[0..3]=1, mins[0..3]=0: q[0..4]=1, q[4..8]=0
        for j in 0..4 {
            buf[4 + j] = 1;
        }
        // For scales[4..7]=1, mins[4..7]=0: need (q[j+4]&0xF)=1, (q[j+4]>>4)=0 → q[8..12]=0x01
        for j in 4..8 {
            buf[4 + j + 4] = 0x01;
        }
        // qs[*] already 0
        buf
    }

    #[test]
    fn q4_k_zero_block_dequants_to_zero() {
        let src = synth_q4_k_zero();
        let mut out = vec![999f32; QK_K];
        dequant_q4_k(&src, &mut out).unwrap();
        for &v in &out {
            assert_eq!(v, 0.0, "dequant of all-zero quants must be zero");
        }
    }

    #[test]
    fn dequant_into_f16_f16_passthrough_is_lossless() {
        // F16 source → identical f16 bits, and decoding them back equals the originals.
        let vals = [1.0f32, 2.0, -0.5, 0.0, 65504.0]; // last is f16::MAX
        let mut src = Vec::new();
        for &v in &vals {
            src.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
        }
        let mut bits = vec![0u16; vals.len()];
        dequant_into_f16(GgmlDtype::F16, &src, &mut bits).unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(bits[i], f16::from_f32(v).to_bits(), "f16 passthrough bit-exact at {i}");
            assert_eq!(f16::from_bits(bits[i]).to_f32(), v, "decoded value matches at {i}");
        }
    }

    #[test]
    fn dequant_into_f16_f32_downcast() {
        // F32 source → f16 bits equal to a direct f32→f16 conversion.
        let vals = [1.0f32, 2.0, -0.5, 0.1];
        let mut src = Vec::new();
        for &v in &vals {
            src.extend_from_slice(&v.to_le_bytes());
        }
        let mut bits = vec![0u16; vals.len()];
        dequant_into_f16(GgmlDtype::F32, &src, &mut bits).unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(bits[i], f16::from_f32(v).to_bits(), "f32→f16 downcast matches at {i}");
        }
    }

    /// Synthetic Q4_K with d=1.0, dmin=0.0, all scales=1 (mins=0), and qs filled with
    /// alternating low/high nibbles 0xA / 0x5. Expect dequant = pattern of 5,10,5,10,…
    #[test]
    fn q4_k_alternating_nibbles() {
        let mut buf = synth_q4_k_zero();
        // qs[i] = 0xA5 → low nibble 5, high nibble 10
        for b in &mut buf[16..16 + 128] {
            *b = 0xA5;
        }
        let mut out = vec![0f32; QK_K];
        dequant_q4_k(&buf, &mut out).unwrap();
        // For each 64-elem chunk: first 32 from low nibbles (=5), next 32 from high (=10)
        for chunk in 0..(QK_K / 64) {
            for l in 0..32 {
                assert_eq!(out[chunk * 64 + l], 5.0, "low nibble dequant");
                assert_eq!(out[chunk * 64 + l + 32], 10.0, "high nibble dequant");
            }
        }
    }

    #[test]
    fn q6_k_zero_block_dequants_to_zero() {
        // d=1.0 (f16), all ql/qh/scales=0: each quant = (0|0)-32 = -32.
        // Wait: scales also 0, so 0*-32 = 0. So output is all zeros.
        let mut buf = vec![0u8; Q6_K_BLOCK_BYTES];
        // d at offset 128+64+16 = 208
        buf[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
        let mut out = vec![999f32; QK_K];
        dequant_q6_k(&buf, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q6_k_unit_scale_constant_quants() {
        // d=1.0, scales[*]=1, ql=0, qh=0 → quant=(0|0)-32=-32 → output = 1*1*(-32)=-32 everywhere.
        let mut buf = vec![0u8; Q6_K_BLOCK_BYTES];
        // scales at offset 128+64 = 192, 16 i8s of value 1
        for i in 0..16 {
            buf[192 + i] = 1;
        }
        // d=1.0 (f16) at offset 208
        buf[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
        let mut out = vec![0f32; QK_K];
        dequant_q6_k(&buf, &mut out).unwrap();
        for &v in &out {
            assert_eq!(v, -32.0);
        }
    }

    #[test]
    fn f16_round_trip() {
        let values: [f32; 4] = [0.0, 1.0, -2.5, 3.5];
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for v in values {
            bytes.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
        }
        let mut out = vec![0f32; values.len()];
        f16_to_f32(&bytes, &mut out).unwrap();
        for i in 0..values.len() {
            assert!(
                (out[i] - values[i]).abs() < 0.01,
                "got {} want {}",
                out[i],
                values[i]
            );
        }
    }
}
