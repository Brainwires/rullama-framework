//! Storage dtypes for Ollama image-model safetensors blobs.
//!
//! Covers the *float* storage formats (`F32` / `F16` / `BF16` / `F8_E4M3` /
//! `F8_E5M2`) plus the integer index types that appear in headers. Grouped /
//! affine quantization (`int4` / `int8` / `nvfp4` / `mxfp8`) is NOT a single
//! tensor dtype — it's a packed weight tensor (often stored as `U8`/`U32`) plus
//! companion `.scale` / `.bias` tensors and a `__metadata__.quant_type`. That
//! reconstruction lives one layer up (in the loader), not here; this enum only
//! describes how to read the raw bytes of one tensor.
//!
//! Reference: Ollama `x/imagegen/safetensors/safetensors.go` dtype table and
//! `x/imagegen/docs/blob-format.md`.

use half::{bf16, f16};

use crate::error::{Result, RullamaError};

/// A safetensors element dtype as written in the JSON header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StDtype {
    F32,
    F16,
    Bf16,
    /// 8-bit float, 1 sign / 4 exp (bias 7) / 3 mantissa, with inf/nan per OCP.
    F8E4M3,
    /// 8-bit float, 1 sign / 5 exp (bias 15) / 2 mantissa.
    F8E5M2,
    U8,
    I8,
    I16,
    I32,
    I64,
    U16,
    U32,
}

impl StDtype {
    /// Parse a safetensors dtype string (accepts the common aliases Ollama's
    /// reader accepts, e.g. `FLOAT32`).
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" | "FLOAT32" => Self::F32,
            "F16" | "FLOAT16" => Self::F16,
            "BF16" | "BFLOAT16" => Self::Bf16,
            "F8_E4M3" | "F8_E4M3FN" => Self::F8E4M3,
            "F8_E5M2" | "F8_E5M2FNUZ" | "F8_E5M2FN" => Self::F8E5M2,
            "U8" | "UINT8" => Self::U8,
            "I8" | "INT8" => Self::I8,
            "I16" | "INT16" => Self::I16,
            "I32" | "INT32" => Self::I32,
            "I64" | "INT64" => Self::I64,
            "U16" | "UINT16" => Self::U16,
            "U32" | "UINT32" => Self::U32,
            other => {
                return Err(RullamaError::Image(format!(
                    "unsupported safetensors dtype {other:?}"
                )));
            }
        })
    }

    /// Bytes per element.
    pub fn elem_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::Bf16 | Self::I16 | Self::U16 => 2,
            Self::F8E4M3 | Self::F8E5M2 | Self::U8 | Self::I8 => 1,
            Self::I64 => 8,
        }
    }

    /// Whether this dtype is a float we can dequantize to f32 directly.
    pub fn is_float(self) -> bool {
        matches!(
            self,
            Self::F32 | Self::F16 | Self::Bf16 | Self::F8E4M3 | Self::F8E5M2
        )
    }

    /// Dequantize a tightly-packed byte slice (`elem_count * elem_size` bytes)
    /// of this dtype into `f32`. Only valid for float dtypes; integer dtypes
    /// return an error (their semantics depend on the quant scheme).
    pub fn dequant_to_f32(self, bytes: &[u8]) -> Result<Vec<f32>> {
        let es = self.elem_size();
        if !bytes.len().is_multiple_of(es) {
            return Err(RullamaError::Image(format!(
                "byte length {} not a multiple of elem size {es}",
                bytes.len()
            )));
        }
        let n = bytes.len() / es;
        let mut out = Vec::with_capacity(n);
        match self {
            Self::F32 => {
                for c in bytes.chunks_exact(4) {
                    out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            Self::F16 => {
                for c in bytes.chunks_exact(2) {
                    out.push(f16::from_le_bytes([c[0], c[1]]).to_f32());
                }
            }
            Self::Bf16 => {
                for c in bytes.chunks_exact(2) {
                    out.push(bf16::from_le_bytes([c[0], c[1]]).to_f32());
                }
            }
            Self::F8E4M3 => {
                for &b in bytes {
                    out.push(f8_e4m3_to_f32(b));
                }
            }
            Self::F8E5M2 => {
                for &b in bytes {
                    out.push(f8_e5m2_to_f32(b));
                }
            }
            _ => {
                return Err(RullamaError::Image(format!(
                    "dequant_to_f32 called on non-float dtype {self:?}"
                )));
            }
        }
        Ok(out)
    }
}

/// Decode an OCP `E4M3` 8-bit float (1-4-3, exponent bias 7).
///
/// E4M3FN convention: no infinities; `S 1111 111` is NaN. Subnormals use a
/// zero exponent field with an implicit-0 leading bit.
fn f8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let mant = (b & 0x07) as i32;
    if exp == 0 {
        // subnormal (or zero): value = mant/8 * 2^(1-bias)
        if mant == 0 {
            return sign * 0.0;
        }
        let m = mant as f32 / 8.0;
        return sign * m * 2f32.powi(1 - 7);
    }
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN; // the single NaN encoding in E4M3FN
    }
    let m = 1.0 + mant as f32 / 8.0;
    sign * m * 2f32.powi(exp - 7)
}

/// Decode an `E5M2` 8-bit float (1-5-2, exponent bias 15). Has inf/nan like a
/// shrunk IEEE half.
fn f8_e5m2_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 2) & 0x1F) as i32;
    let mant = (b & 0x03) as i32;
    if exp == 0 {
        if mant == 0 {
            return sign * 0.0;
        }
        let m = mant as f32 / 4.0;
        return sign * m * 2f32.powi(1 - 15);
    }
    if exp == 0x1F {
        return if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    let m = 1.0 + mant as f32 / 4.0;
    sign * m * 2f32.powi(exp - 15)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_aliases_and_sizes() {
        assert_eq!(StDtype::parse("BFLOAT16").unwrap(), StDtype::Bf16);
        assert_eq!(StDtype::parse("F8_E4M3").unwrap(), StDtype::F8E4M3);
        assert_eq!(StDtype::Bf16.elem_size(), 2);
        assert_eq!(StDtype::F8E5M2.elem_size(), 1);
        assert!(StDtype::parse("Q4_K").is_err());
    }

    #[test]
    fn dequant_f32_roundtrip() {
        let vals = [1.0f32, -2.5, 0.0, 1234.5];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(StDtype::F32.dequant_to_f32(&bytes).unwrap(), vals);
    }

    #[test]
    fn dequant_bf16_and_f16() {
        let v = 3.5f32;
        let bf = bf16::from_f32(v);
        assert_eq!(
            StDtype::Bf16.dequant_to_f32(&bf.to_le_bytes()).unwrap()[0],
            bf.to_f32()
        );
        let h = f16::from_f32(v);
        assert_eq!(
            StDtype::F16.dequant_to_f32(&h.to_le_bytes()).unwrap()[0],
            h.to_f32()
        );
    }

    #[test]
    fn f8_e4m3_known_values() {
        // 0x00 = +0, 0x80 = -0
        assert_eq!(f8_e4m3_to_f32(0x00), 0.0);
        // exp=7 (bias) mant=0 → 1.0  → bits: 0 0111 000 = 0x38
        assert_eq!(f8_e4m3_to_f32(0x38), 1.0);
        // 2.0 → 1.0 * 2^1 → exp=8 mant=0 → 0 1000 000 = 0x40
        assert_eq!(f8_e4m3_to_f32(0x40), 2.0);
        // -1.0 → 0xB8
        assert_eq!(f8_e4m3_to_f32(0xB8), -1.0);
        // NaN encoding
        assert!(f8_e4m3_to_f32(0x7F).is_nan());
    }

    #[test]
    fn f8_e5m2_known_values() {
        assert_eq!(f8_e5m2_to_f32(0x00), 0.0);
        // 1.0 → exp=15 mant=0 → 0 01111 00 = 0x3C
        assert_eq!(f8_e5m2_to_f32(0x3C), 1.0);
        // 2.0 → exp=16 → 0 10000 00 = 0x40
        assert_eq!(f8_e5m2_to_f32(0x40), 2.0);
        // +inf → 0 11111 00 = 0x7C
        assert_eq!(f8_e5m2_to_f32(0x7C), f32::INFINITY);
        assert!(f8_e5m2_to_f32(0x7D).is_nan());
    }
}
