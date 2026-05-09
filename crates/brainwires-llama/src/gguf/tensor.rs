//! Tensor accessors backed by a [`GgufReader`].
//!
//! On the CPU reference path we materialize whole tensors as `Vec<f32>`. On the wgpu
//! path (M2+) we'll instead stream raw bytes into GPU storage buffers and dequantize
//! inside a fused matmul kernel — but we still want this code path for the parity oracle.

use crate::error::{Result, RullamaError};
use super::dtype::GgmlDtype;
use super::quant::dequant_into_f32;
use super::reader::GgufReader;

/// Dequantize the named tensor into a freshly-allocated `Vec<f32>`.
///
/// The vector is laid out in the same element order GGUF stores: `dims[0]` is the
/// fastest-varying axis. For a `[k, n]` weight, that means the row of length `k` is
/// contiguous in memory.
pub fn dequant_tensor_to_f32(r: &GgufReader, name: &str) -> Result<Vec<f32>> {
    let desc = r.tensor(name)?;
    let bytes = r.tensor_bytes(name)?;
    let elems = desc.elem_count() as usize;
    let mut out = vec![0f32; elems];
    dequant_into_f32(desc.dtype, bytes, &mut out)?;
    Ok(out)
}

/// Dequantize a single contiguous *row* (slice along the second axis) of a 2-D tensor.
///
/// `row_len` is the size of `dims[0]` (the fastest-varying axis). The row at index
/// `row_idx` occupies elements `[row_idx*row_len .. (row_idx+1)*row_len]` of the
/// (logical) f32 layout. For block-quantized types we require the row to be aligned
/// to the block boundary along `dims[0]` (Gemma 4's row sizes are all multiples of 256
/// along the leading axis, so this always holds).
pub fn dequant_row_to_f32(r: &GgufReader, name: &str, row_idx: usize) -> Result<Vec<f32>> {
    let desc = r.tensor(name)?;
    if desc.dims.len() != 2 {
        return Err(RullamaError::Gguf(format!(
            "dequant_row_to_f32: tensor {} has {} dims, expected 2",
            desc.name, desc.dims.len()
        )));
    }
    let row_len = desc.dims[0] as usize;
    let n_rows = desc.dims[1] as usize;
    if row_idx >= n_rows {
        return Err(RullamaError::Gguf(format!(
            "row {row_idx} out of bounds for tensor {} ({} rows)", desc.name, n_rows
        )));
    }

    let block_elems = desc.dtype.block_elems();
    if row_len % block_elems != 0 {
        return Err(RullamaError::Gguf(format!(
            "row_len {} not multiple of block_elems {} for {}",
            row_len, block_elems, desc.name
        )));
    }
    let blocks_per_row = row_len / block_elems;
    let bytes_per_row = blocks_per_row * desc.dtype.block_bytes();

    let all_bytes = r.tensor_bytes(name)?;
    let start = row_idx * bytes_per_row;
    let end = start + bytes_per_row;
    if end > all_bytes.len() {
        return Err(RullamaError::Gguf(format!(
            "row bytes {start}..{end} extend past tensor data {} for {}",
            all_bytes.len(), desc.name
        )));
    }

    let mut out = vec![0f32; row_len];
    dequant_into_f32(desc.dtype, &all_bytes[start..end], &mut out)?;
    Ok(out)
}

/// Convenience: known dtype helper.
#[allow(dead_code)]
pub(crate) fn dtype_of(r: &GgufReader, name: &str) -> Result<GgmlDtype> {
    Ok(r.tensor(name)?.dtype)
}

// ---------- async (streaming-safe) variants ----------
//
// The streaming reader can't return a borrow, so the dequant has to take owned bytes.
// On an in-memory reader these are equivalent to the sync variants plus one memcpy.

/// Async equivalent of [`dequant_tensor_to_f32`]. Works for both in-memory and
/// streaming readers. The fetched bytes are dropped before this returns.
pub async fn dequant_tensor_to_f32_async(r: &GgufReader, name: &str) -> Result<Vec<f32>> {
    let desc = r.tensor(name)?.clone();
    let bytes = r.fetch_tensor_bytes(name).await?;
    let elems = desc.elem_count() as usize;
    let mut out = vec![0f32; elems];
    dequant_into_f32(desc.dtype, &bytes, &mut out)?;
    Ok(out)
}

/// Async equivalent of [`dequant_row_to_f32`]. Fetches only the row's bytes when the
/// underlying fetcher supports byte-range reads (e.g. HTTP Range).
pub async fn dequant_row_to_f32_async(r: &GgufReader, name: &str, row_idx: usize) -> Result<Vec<f32>> {
    let desc = r.tensor(name)?.clone();
    if desc.dims.len() != 2 {
        return Err(RullamaError::Gguf(format!(
            "dequant_row_to_f32_async: tensor {} has {} dims, expected 2",
            desc.name, desc.dims.len()
        )));
    }
    let row_len = desc.dims[0] as usize;
    let n_rows = desc.dims[1] as usize;
    if row_idx >= n_rows {
        return Err(RullamaError::Gguf(format!(
            "row {row_idx} out of bounds for tensor {} ({} rows)", desc.name, n_rows
        )));
    }

    let block_elems = desc.dtype.block_elems();
    if row_len % block_elems != 0 {
        return Err(RullamaError::Gguf(format!(
            "row_len {} not multiple of block_elems {} for {}",
            row_len, block_elems, desc.name
        )));
    }
    let blocks_per_row = row_len / block_elems;
    let bytes_per_row = blocks_per_row * desc.dtype.block_bytes();

    // Fetch only the row's bytes via the fetcher (Range request when streaming).
    let row_bytes = {
        let abs_offset = (r.tensor(name)?.offset
            + (row_idx * bytes_per_row) as u64)
            + r.data_section_offset();
        r.fetcher().fetch(abs_offset, bytes_per_row as u64).await?
    };

    let mut out = vec![0f32; row_len];
    dequant_into_f32(desc.dtype, &row_bytes, &mut out)?;
    Ok(out)
}
