//! Tensor accessors backed by a [`GgufReader`].
//!
//! On the CPU reference path we materialize whole tensors as `Vec<f32>`. On the wgpu
//! path (M2+) we'll instead stream raw bytes into GPU storage buffers and dequantize
//! inside a fused matmul kernel — but we still want this code path for the parity oracle.

use super::dtype::GgmlDtype;
use super::quant::{dequant_into_f16, dequant_into_f32};
use super::reader::GgufReader;
use crate::error::{Result, RullamaError};

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
            desc.name,
            desc.dims.len()
        )));
    }
    let row_len = desc.dims[0] as usize;
    let n_rows = desc.dims[1] as usize;
    if row_idx >= n_rows {
        return Err(RullamaError::Gguf(format!(
            "row {row_idx} out of bounds for tensor {} ({} rows)",
            desc.name, n_rows
        )));
    }

    let block_elems = desc.dtype.block_elems();
    if !row_len.is_multiple_of(block_elems) {
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
            all_bytes.len(),
            desc.name
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

/// Byte extent of one expert's 2-D slice inside a 3-D stacked expert tensor
/// (`ffn_*_exps.weight`, dims `[in, out, n_experts]` — expert is the
/// slowest-varying axis, so each expert's `[in, out]` matrix is contiguous).
/// Returns `(slice_bytes, in_len, out_len)`.
fn expert_slice_extent(desc: &super::reader::TensorDesc) -> Result<(usize, usize, usize)> {
    if desc.dims.len() != 3 {
        return Err(RullamaError::Gguf(format!(
            "expert slice: tensor {} has {} dims, expected 3",
            desc.name,
            desc.dims.len()
        )));
    }
    let in_len = desc.dims[0] as usize;
    let out_len = desc.dims[1] as usize;
    let block_elems = desc.dtype.block_elems();
    if !in_len.is_multiple_of(block_elems) {
        return Err(RullamaError::Gguf(format!(
            "expert slice: in_len {} not multiple of block_elems {} for {}",
            in_len, block_elems, desc.name
        )));
    }
    let bytes_per_row = (in_len / block_elems) * desc.dtype.block_bytes();
    Ok((bytes_per_row * out_len, in_len, out_len))
}

/// Dequantize a single expert's contiguous 2-D `[in, out]` slice out of a 3-D
/// stacked MoE expert tensor (`blk.N.ffn_{gate,up,down,gate_up}_exps.weight`,
/// dims `[in, out, n_experts]`). Quant blocks never straddle rows (row sizes
/// are block-aligned), so the slice window reuses the plain 2-D dequant.
pub fn dequant_expert_slice_to_f32(
    r: &GgufReader,
    name: &str,
    expert_idx: usize,
) -> Result<Vec<f32>> {
    let desc = r.tensor(name)?;
    let (slice_bytes, in_len, out_len) = expert_slice_extent(desc)?;
    let n_experts = desc.dims[2] as usize;
    if expert_idx >= n_experts {
        return Err(RullamaError::Gguf(format!(
            "expert {expert_idx} out of bounds for tensor {} ({} experts)",
            desc.name, n_experts
        )));
    }
    let all_bytes = r.tensor_bytes(name)?;
    let start = expert_idx * slice_bytes;
    let end = start + slice_bytes;
    if end > all_bytes.len() {
        return Err(RullamaError::Gguf(format!(
            "expert bytes {start}..{end} extend past tensor data {} for {}",
            all_bytes.len(),
            desc.name
        )));
    }
    let mut out = vec![0f32; in_len * out_len];
    dequant_into_f32(desc.dtype, &all_bytes[start..end], &mut out)?;
    Ok(out)
}

/// Async equivalent of [`dequant_expert_slice_to_f32`]. Fetches only the
/// expert's byte window when the fetcher supports range reads.
pub async fn dequant_expert_slice_to_f32_async(
    r: &GgufReader,
    name: &str,
    expert_idx: usize,
) -> Result<Vec<f32>> {
    let desc = r.tensor(name)?.clone();
    let (slice_bytes, in_len, out_len) = expert_slice_extent(&desc)?;
    let n_experts = desc.dims[2] as usize;
    if expert_idx >= n_experts {
        return Err(RullamaError::Gguf(format!(
            "expert {expert_idx} out of bounds for tensor {} ({} experts)",
            desc.name, n_experts
        )));
    }
    let slice = {
        let abs_offset = desc.offset + (expert_idx * slice_bytes) as u64 + r.data_section_offset();
        r.fetcher().fetch(abs_offset, slice_bytes as u64).await?
    };
    let mut out = vec![0f32; in_len * out_len];
    dequant_into_f32(desc.dtype, &slice, &mut out)?;
    Ok(out)
}

/// Dequantize the named tensor into a freshly-allocated `Vec<u16>` of raw f16
/// bit patterns (little-endian half layout). F16 tensors pass through
/// losslessly; other dtypes are dequantized to f32 then downcast. Used by the
/// f16-resident StyleTTS2 path to hold big weights at half the host footprint.
pub fn dequant_tensor_to_f16(r: &GgufReader, name: &str) -> Result<Vec<u16>> {
    let desc = r.tensor(name)?;
    let bytes = r.tensor_bytes(name)?;
    let elems = desc.elem_count() as usize;
    let mut out = vec![0u16; elems];
    dequant_into_f16(desc.dtype, bytes, &mut out)?;
    Ok(out)
}

/// Async equivalent of [`dequant_tensor_to_f16`] for the streaming loader.
pub async fn dequant_tensor_to_f16_async(r: &GgufReader, name: &str) -> Result<Vec<u16>> {
    let desc = r.tensor(name)?.clone();
    let bytes = r.fetch_tensor_bytes(name).await?;
    let elems = desc.elem_count() as usize;
    let mut out = vec![0u16; elems];
    dequant_into_f16(desc.dtype, &bytes, &mut out)?;
    Ok(out)
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

/// Test-support: a `GgufReader` over only the HEADER of an on-disk GGUF
/// (metadata + tensor descs — the first 64 MB). Reading a whole 7+ GB fixture
/// blob into a Vec just to look up metadata or one tensor peaks gigabytes of
/// resident memory per test and OOMs the suite on 16 GB machines. Tensor DATA
/// is not in the returned reader — pair with [`read_tensor_raw`].
#[cfg(test)]
pub(crate) fn reader_from_file_header(path: &str) -> Result<GgufReader> {
    use std::io::Read;
    let mut f =
        std::fs::File::open(path).map_err(|e| RullamaError::Gguf(format!("open {path}: {e}")))?;
    let len = f
        .metadata()
        .map_err(|e| RullamaError::Gguf(format!("stat {path}: {e}")))?
        .len();
    let take = len.min(64 * 1024 * 1024) as usize;
    let mut header = vec![0u8; take];
    f.read_exact(&mut header)
        .map_err(|e| RullamaError::Gguf(format!("read {path}: {e}")))?;
    GgufReader::new(header)
}

/// Test-support: fetch one tensor's raw (still-quantized) bytes straight from
/// the file via seek + exact read. `r` must come from [`reader_from_file_header`]
/// on the same path.
#[cfg(test)]
pub(crate) fn read_tensor_raw(path: &str, r: &GgufReader, name: &str) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let desc = r.tensor(name)?.clone();
    let block_elems = desc.dtype.block_elems();
    let n_blocks = (desc.elem_count() as usize).div_ceil(block_elems);
    let n_bytes = n_blocks * desc.dtype.block_bytes();
    let mut f =
        std::fs::File::open(path).map_err(|e| RullamaError::Gguf(format!("open {path}: {e}")))?;
    f.seek(SeekFrom::Start(r.data_section_offset() + desc.offset))
        .map_err(|e| RullamaError::Gguf(format!("seek {path}: {e}")))?;
    let mut raw = vec![0u8; n_bytes];
    f.read_exact(&mut raw)
        .map_err(|e| RullamaError::Gguf(format!("read {name} from {path}: {e}")))?;
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::quant::{Q8_0_BLOCK_BYTES, QK8_0};

    /// Minimal GGUF v3 writer — just enough for unit-testing tensor accessors
    /// without a multi-GB model blob (the T0 "synthetic fixture" lever).
    /// No metadata KVs (alignment defaults to 32).
    fn synth_gguf(tensors: &[(&str, &[u64], GgmlDtype, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // 'GGUF'
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // no metadata KVs

        // Tensor infos. Offsets are relative to the (32-aligned) data section
        // and themselves 32-aligned.
        let mut data_off = 0u64;
        let mut offsets = Vec::new();
        for (name, dims, dtype, bytes) in tensors {
            out.extend_from_slice(&(name.len() as u64).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&(*dtype as u32).to_le_bytes());
            out.extend_from_slice(&data_off.to_le_bytes());
            offsets.push(data_off);
            data_off = (data_off + bytes.len() as u64).div_ceil(32) * 32;
        }
        // Pad header to the 32-byte data-section boundary, then write data.
        while !out.len().is_multiple_of(32) {
            out.push(0);
        }
        let data_start = out.len();
        for ((_, _, _, bytes), off) in tensors.iter().zip(&offsets) {
            let want = data_start + *off as usize;
            while out.len() < want {
                out.push(0);
            }
            out.extend_from_slice(bytes);
        }
        out
    }

    /// One synthetic Q8_0 block: scale d (f16) + 32 deterministic int8 quants
    /// seeded by `seed`.
    fn q8_0_block(d: f32, seed: usize) -> Vec<u8> {
        let mut b = vec![0u8; Q8_0_BLOCK_BYTES];
        b[0..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for l in 0..QK8_0 {
            b[2 + l] = ((seed * 37 + l * 13) % 256) as u8;
        }
        b
    }

    /// A 3-D stacked expert tensor sliced per expert must dequant to exactly
    /// the corresponding window of the whole-tensor dequant.
    #[test]
    fn expert_slice_matches_whole_tensor_window() {
        // dims [in=64, out=3, n_experts=4] in Q8_0: 2 blocks/row, 3 rows/expert.
        let (in_len, out_len, n_exp) = (64usize, 3usize, 4usize);
        let blocks_per_row = in_len / QK8_0;
        let mut bytes = Vec::new();
        for e in 0..n_exp {
            for row in 0..out_len {
                for blk in 0..blocks_per_row {
                    bytes.extend_from_slice(&q8_0_block(
                        0.5 + e as f32 * 0.25,
                        e * 100 + row * 10 + blk,
                    ));
                }
            }
        }
        let file = synth_gguf(&[(
            "blk.0.ffn_gate_exps.weight",
            &[in_len as u64, out_len as u64, n_exp as u64],
            GgmlDtype::Q8_0,
            bytes,
        )]);
        let r = GgufReader::new(file).expect("synth gguf parses");

        let whole = dequant_tensor_to_f32(&r, "blk.0.ffn_gate_exps.weight").unwrap();
        assert_eq!(whole.len(), in_len * out_len * n_exp);
        for e in 0..n_exp {
            let slice = dequant_expert_slice_to_f32(&r, "blk.0.ffn_gate_exps.weight", e).unwrap();
            assert_eq!(slice.len(), in_len * out_len);
            let window = &whole[e * in_len * out_len..(e + 1) * in_len * out_len];
            assert_eq!(slice, window, "expert {e} slice != whole-tensor window");
            // async variant agrees
            let aslice = pollster::block_on(dequant_expert_slice_to_f32_async(
                &r,
                "blk.0.ffn_gate_exps.weight",
                e,
            ))
            .unwrap();
            assert_eq!(aslice, slice, "expert {e} async != sync");
        }
    }

    #[test]
    fn expert_slice_rejects_2d_and_oob() {
        let bytes = q8_0_block(1.0, 0);
        let file = synth_gguf(&[
            ("two_d", &[32u64, 1], GgmlDtype::Q8_0, bytes.clone()),
            ("exps", &[32u64, 1, 1], GgmlDtype::Q8_0, bytes),
        ]);
        let r = GgufReader::new(file).unwrap();
        assert!(dequant_expert_slice_to_f32(&r, "two_d", 0).is_err());
        assert!(dequant_expert_slice_to_f32(&r, "exps", 1).is_err()); // only 1 expert
        assert!(dequant_expert_slice_to_f32(&r, "exps", 0).is_ok());
    }
}

/// Async equivalent of [`dequant_row_to_f32`]. Fetches only the row's bytes when the
/// underlying fetcher supports byte-range reads (e.g. HTTP Range).
pub async fn dequant_row_to_f32_async(
    r: &GgufReader,
    name: &str,
    row_idx: usize,
) -> Result<Vec<f32>> {
    let desc = r.tensor(name)?.clone();
    if desc.dims.len() != 2 {
        return Err(RullamaError::Gguf(format!(
            "dequant_row_to_f32_async: tensor {} has {} dims, expected 2",
            desc.name,
            desc.dims.len()
        )));
    }
    let row_len = desc.dims[0] as usize;
    let n_rows = desc.dims[1] as usize;
    if row_idx >= n_rows {
        return Err(RullamaError::Gguf(format!(
            "row {row_idx} out of bounds for tensor {} ({} rows)",
            desc.name, n_rows
        )));
    }

    let block_elems = desc.dtype.block_elems();
    if !row_len.is_multiple_of(block_elems) {
        return Err(RullamaError::Gguf(format!(
            "row_len {} not multiple of block_elems {} for {}",
            row_len, block_elems, desc.name
        )));
    }
    let blocks_per_row = row_len / block_elems;
    let bytes_per_row = blocks_per_row * desc.dtype.block_bytes();

    // Fetch only the row's bytes via the fetcher (Range request when streaming).
    let row_bytes = {
        let abs_offset =
            (r.tensor(name)?.offset + (row_idx * bytes_per_row) as u64) + r.data_section_offset();
        r.fetcher().fetch(abs_offset, bytes_per_row as u64).await?
    };

    let mut out = vec![0f32; row_len];
    dequant_into_f32(desc.dtype, &row_bytes, &mut out)?;
    Ok(out)
}
