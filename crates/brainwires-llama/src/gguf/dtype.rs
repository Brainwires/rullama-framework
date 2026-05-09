//! GGML tensor dtypes. The integer codes match `enum ggml_type` in ggml.h. We only
//! enumerate the types we expect to encounter; unknown codes are surfaced as an error
//! rather than silently mapped.

use crate::error::{Result, RullamaError};

/// GGML tensor element type. Integer codes are stable in the GGUF format.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlDtype {
    F32   = 0,
    F16   = 1,
    Q4_0  = 2,
    Q4_1  = 3,
    Q5_0  = 6,
    Q5_1  = 7,
    Q8_0  = 8,
    Q8_1  = 9,
    Q2_K  = 10,
    Q3_K  = 11,
    Q4_K  = 12,
    Q5_K  = 13,
    Q6_K  = 14,
    Q8_K  = 15,
    I8    = 16,
    I16   = 17,
    I32   = 18,
    I64   = 19,
    F64   = 20,
    BF16  = 30,
}

impl GgmlDtype {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::F32, 1 => Self::F16,
            2 => Self::Q4_0, 3 => Self::Q4_1,
            6 => Self::Q5_0, 7 => Self::Q5_1,
            8 => Self::Q8_0, 9 => Self::Q8_1,
            10 => Self::Q2_K, 11 => Self::Q3_K, 12 => Self::Q4_K, 13 => Self::Q5_K,
            14 => Self::Q6_K, 15 => Self::Q8_K,
            16 => Self::I8, 17 => Self::I16, 18 => Self::I32, 19 => Self::I64,
            20 => Self::F64, 30 => Self::BF16,
            other => return Err(RullamaError::Gguf(format!("unknown ggml dtype {other}"))),
        })
    }

    /// Number of elements per quantization block. `1` for unquantized types.
    pub fn block_elems(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64
            | Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1
            | Self::Q8_0 | Self::Q8_1 => 32,
            // K-quants
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K
            | Self::Q6_K | Self::Q8_K => 256,
        }
    }

    /// Number of bytes per quantization block. Source: `ggml_type_size` in ggml.h.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1, Self::I16 => 2, Self::I32 => 4, Self::I64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
        }
    }
}
