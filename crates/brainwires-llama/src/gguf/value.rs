//! GGUF metadata value types. Mirror the on-wire `gguf_metadata_value_type` enum.

use crate::error::{Result, RullamaError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    U8 = 0, I8 = 1,
    U16 = 2, I16 = 3,
    U32 = 4, I32 = 5,
    F32 = 6, Bool = 7,
    String = 8, Array = 9,
    U64 = 10, I64 = 11,
    F64 = 12,
}

impl GgufValueType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::U8, 1 => Self::I8,
            2 => Self::U16, 3 => Self::I16,
            4 => Self::U32, 5 => Self::I32,
            6 => Self::F32, 7 => Self::Bool,
            8 => Self::String, 9 => Self::Array,
            10 => Self::U64, 11 => Self::I64, 12 => Self::F64,
            other => return Err(RullamaError::Gguf(format!("unknown gguf value type {other}"))),
        })
    }
}

/// A decoded GGUF metadata value.
///
/// Arrays of scalars are eagerly decoded into a flat typed `Vec`. Arrays of strings (the
/// vocab is the big one — 262K entries on Gemma 4) are also eagerly decoded; the cost is
/// a few MB and the simplicity is worth it for now.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8), I8(i8),
    U16(u16), I16(i16),
    U32(u32), I32(i32),
    U64(u64), I64(i64),
    F32(f32), F64(f64),
    Bool(bool),
    String(String),
    ArrayU8(Vec<u8>),  ArrayI8(Vec<i8>),
    ArrayU16(Vec<u16>), ArrayI16(Vec<i16>),
    ArrayU32(Vec<u32>), ArrayI32(Vec<i32>),
    ArrayU64(Vec<u64>), ArrayI64(Vec<i64>),
    ArrayF32(Vec<f32>), ArrayF64(Vec<f64>),
    ArrayBool(Vec<bool>),
    ArrayString(Vec<String>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Result<u32> {
        match self {
            Self::U32(v) => Ok(*v),
            Self::U64(v) => u32::try_from(*v).map_err(|_| RullamaError::Gguf(format!("u64→u32 overflow: {v}"))),
            Self::I32(v) => u32::try_from(*v).map_err(|_| RullamaError::Gguf(format!("i32 negative: {v}"))),
            Self::I64(v) => u32::try_from(*v).map_err(|_| RullamaError::Gguf(format!("i64 negative or overflow: {v}"))),
            other => Err(RullamaError::Gguf(format!("expected u32, got {other:?}"))),
        }
    }

    pub fn as_u64(&self) -> Result<u64> {
        match self {
            Self::U64(v) => Ok(*v),
            Self::U32(v) => Ok(*v as u64),
            Self::I64(v) => u64::try_from(*v).map_err(|_| RullamaError::Gguf(format!("i64 negative: {v}"))),
            Self::I32(v) => u64::try_from(*v).map_err(|_| RullamaError::Gguf(format!("i32 negative: {v}"))),
            other => Err(RullamaError::Gguf(format!("expected u64, got {other:?}"))),
        }
    }

    pub fn as_f32(&self) -> Result<f32> {
        match self {
            Self::F32(v) => Ok(*v),
            Self::F64(v) => Ok(*v as f32),
            other => Err(RullamaError::Gguf(format!("expected f32, got {other:?}"))),
        }
    }

    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            other => Err(RullamaError::Gguf(format!("expected bool, got {other:?}"))),
        }
    }

    pub fn as_str(&self) -> Result<&str> {
        match self {
            Self::String(s) => Ok(s.as_str()),
            other => Err(RullamaError::Gguf(format!("expected string, got {other:?}"))),
        }
    }

    pub fn as_string_array(&self) -> Result<&[String]> {
        match self {
            Self::ArrayString(v) => Ok(v.as_slice()),
            other => Err(RullamaError::Gguf(format!("expected string array, got {other:?}"))),
        }
    }

    pub fn as_u32_array(&self) -> Result<Vec<u32>> {
        Ok(match self {
            Self::ArrayU32(v) => v.clone(),
            Self::ArrayI32(v) => v.iter().map(|x| *x as u32).collect(),
            Self::ArrayU64(v) => v.iter().map(|x| *x as u32).collect(),
            Self::ArrayI64(v) => v.iter().map(|x| *x as u32).collect(),
            other => return Err(RullamaError::Gguf(format!("expected u32 array, got {other:?}"))),
        })
    }

    pub fn as_bool_array(&self) -> Result<&[bool]> {
        match self {
            Self::ArrayBool(v) => Ok(v.as_slice()),
            other => Err(RullamaError::Gguf(format!("expected bool array, got {other:?}"))),
        }
    }
}
