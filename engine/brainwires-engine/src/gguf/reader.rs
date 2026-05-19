//! Streaming GGUF v3 byte-buffer reader.
//!
//! Layout:
//!   magic("GGUF") | version(u32) | tensor_count(u64) | metadata_kv_count(u64)
//!   metadata_kv * metadata_kv_count
//!   tensor_info * tensor_count
//!   alignment padding (general.alignment, default 32)
//!   tensor_data
//!
//! `metadata_kv`  = string key + (value_type, value)
//! `tensor_info`  = string name + n_dims(u32) + dimensions(u64*n_dims) + dtype(u32) + offset(u64)
//!
//! Strings: u64 length + utf-8 bytes.
//!
//! M6 split: the reader keeps the parsed header (5–10 MB of metadata + tensor descriptors)
//! plus a [`TensorFetcher`] for the bulk tensor bytes. The in-memory path keeps an
//! `Arc<[u8]>` so existing sync callers can borrow tensor bytes without copying. The
//! streaming path drops bytes after each tensor reaches the GPU.

use std::collections::HashMap;
use std::sync::Arc;

use super::dtype::GgmlDtype;
use super::fetcher::{InMemoryFetcher, TensorFetcher};
use super::value::{GgufValue, GgufValueType};
use crate::error::{Result, RullamaError};

const GGUF_MAGIC: u32 = 0x4655_4747; // 'GGUF' little-endian
const SUPPORTED_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;

/// Initial header read for streaming readers. The header holds metadata + tensor
/// descriptors; in practice this fits well under 32 MiB even for E4B-class GGUFs.
/// If the header turns out to be larger we re-fetch with the precise length.
const STREAMING_HEADER_HINT: u64 = 32 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgmlDtype,
    /// Byte offset relative to the start of the tensor-data section (not the whole file).
    pub offset: u64,
}

impl TensorDesc {
    /// Total number of elements (product of dims).
    pub fn elem_count(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Total byte length of this tensor in the data section.
    pub fn byte_len(&self) -> u64 {
        let elems = self.elem_count() as usize;
        let block_elems = self.dtype.block_elems();
        let blocks = elems / block_elems;
        debug_assert!(
            elems.is_multiple_of(block_elems),
            "tensor {} has {} elems, not a multiple of block_elems {}",
            self.name,
            elems,
            block_elems
        );
        (blocks * self.dtype.block_bytes()) as u64
    }
}

/// Decoded view of a GGUF byte buffer.
///
/// In-memory readers keep an `Arc<[u8]>` so `tensor_bytes()` returns a borrow at zero copy
/// cost. Streaming readers (`new_streaming`) hold only a `dyn TensorFetcher` and drop
/// bytes after each tensor is consumed; on those, `tensor_bytes()` errors and callers
/// must use `fetch_tensor_bytes().await`.
pub struct GgufReader {
    /// Always present. For in-memory readers this owns the full file. For streaming
    /// readers it owns only the header bytes (used for metadata reads if needed).
    fetcher: Arc<dyn TensorFetcher>,
    /// Set only for in-memory readers. When set, points at the same bytes the fetcher
    /// wraps so callers can borrow tensor bytes synchronously.
    in_memory: Option<Arc<[u8]>>,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorDesc>,
    /// Absolute byte offset where tensor data begins. Tensor `offset` fields are relative to this.
    data_offset: usize,
    alignment: u64,
    version: u32,
}

impl GgufReader {
    /// Build a reader that owns the entire file in memory. Sync; suitable for tests
    /// and native callers where the whole GGUF already fits in RAM.
    ///
    /// Backwards compatible with the pre-M6 API: existing callers using
    /// `GgufReader::new(bytes)` keep working unchanged.
    pub fn new(data: Vec<u8>) -> Result<Self> {
        let arc: Arc<[u8]> = data.into();
        let header = parse_header(&arc)?;
        Ok(Self {
            fetcher: Arc::new(InMemoryFetcher::from_arc(arc.clone())),
            in_memory: Some(arc),
            metadata: header.metadata,
            tensors: header.tensors,
            data_offset: header.data_offset,
            alignment: header.alignment,
            version: header.version,
        })
    }

    /// Build a reader that fetches tensor bytes on demand from a (possibly remote)
    /// source. Pulls a header chunk, parses it, and keeps only the parsed header
    /// resident — tensor bytes are fetched per-call via `fetch_tensor_bytes`.
    pub async fn new_streaming(fetcher: Arc<dyn TensorFetcher>) -> Result<Self> {
        let total = fetcher.total_len();
        let header_len = STREAMING_HEADER_HINT.min(total);
        let mut header_bytes = fetcher.fetch(0, header_len).await?;

        // If the header turns out to extend past our hint (unlikely for current models
        // but possible if vocab arrays balloon), grow the fetch until parsing succeeds.
        loop {
            match parse_header(&header_bytes) {
                Ok(h) => {
                    return Ok(Self {
                        fetcher,
                        in_memory: None,
                        metadata: h.metadata,
                        tensors: h.tensors,
                        data_offset: h.data_offset,
                        alignment: h.alignment,
                        version: h.version,
                    });
                }
                Err(RullamaError::Gguf(msg)) if msg.starts_with("unexpected EOF") => {
                    let new_len = ((header_bytes.len() as u64) * 2).min(total);
                    if new_len == header_bytes.len() as u64 {
                        return Err(RullamaError::Gguf(format!(
                            "header parse failed even after reading the whole file: {msg}"
                        )));
                    }
                    header_bytes = fetcher.fetch(0, new_len).await?;
                }
                Err(e) => return Err(e),
            }
        }
    }

    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn alignment(&self) -> u64 {
        self.alignment
    }
    pub fn metadata(&self) -> &HashMap<String, GgufValue> {
        &self.metadata
    }
    pub fn tensors(&self) -> &[TensorDesc] {
        &self.tensors
    }
    /// Absolute byte offset (from file start) where the tensor-data section begins.
    /// Tensor `offset` fields are relative to this.
    pub fn data_section_offset(&self) -> u64 {
        self.data_offset as u64
    }

    /// Whether this reader has the full file resident in memory (i.e. `tensor_bytes`
    /// can return a borrow). False for streaming readers.
    pub fn is_in_memory(&self) -> bool {
        self.in_memory.is_some()
    }

    /// The fetcher backing this reader. Cloneable; useful for handing async tensor
    /// access to the GPU upload path.
    pub fn fetcher(&self) -> Arc<dyn TensorFetcher> {
        self.fetcher.clone()
    }

    /// Lookup a metadata value by key.
    pub fn get(&self, key: &str) -> Result<&GgufValue> {
        self.metadata
            .get(key)
            .ok_or_else(|| RullamaError::Gguf(format!("missing metadata key: {key}")))
    }

    /// Optional metadata value (returns Ok(None) if the key is absent).
    pub fn get_opt(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Find a tensor descriptor by name.
    pub fn tensor(&self, name: &str) -> Result<&TensorDesc> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| RullamaError::Gguf(format!("missing tensor: {name}")))
    }

    /// Absolute byte range (from file start) for a tensor's payload.
    fn tensor_range(&self, name: &str) -> Result<(u64, u64)> {
        let t = self.tensor(name)?;
        let start = self.data_offset as u64 + t.offset;
        let len = t.byte_len();
        Ok((start, len))
    }

    /// Borrowed bytes for a tensor's payload. Only works for in-memory readers; on a
    /// streaming reader returns an error directing the caller to `fetch_tensor_bytes`.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let bytes = self.in_memory.as_ref().ok_or_else(|| {
            RullamaError::Gguf(format!(
                "tensor_bytes({name}): reader is streaming; use fetch_tensor_bytes().await"
            ))
        })?;
        let (start, len) = self.tensor_range(name)?;
        let s = start as usize;
        let e = s + len as usize;
        if e > bytes.len() {
            return Err(RullamaError::Gguf(format!(
                "tensor {name} extends past buffer end ({e} > {})",
                bytes.len()
            )));
        }
        Ok(&bytes[s..e])
    }

    /// Owned bytes for a tensor's payload, fetched via the underlying [`TensorFetcher`].
    /// Works for both in-memory and streaming readers.
    pub async fn fetch_tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let (start, len) = self.tensor_range(name)?;
        self.fetcher.fetch(start, len).await
    }

    /// Owned bytes for a sub-range of a tensor's payload. `byte_offset` and `byte_len`
    /// are relative to the tensor's payload start (not the file). Useful for streaming
    /// huge tensors (e.g. the 315 MiB `token_embd.weight`) tile-by-tile without ever
    /// materializing the whole thing in wasm linear memory — which is what kills the
    /// WebContent process on iPhone-class shared-RAM devices.
    pub async fn fetch_tensor_range(
        &self,
        name: &str,
        byte_offset: u64,
        byte_len: u64,
    ) -> Result<Vec<u8>> {
        let (start, total) = self.tensor_range(name)?;
        let end = byte_offset.checked_add(byte_len).ok_or_else(|| {
            RullamaError::Gguf(format!(
                "fetch_tensor_range({name}): range overflow {byte_offset}+{byte_len}"
            ))
        })?;
        if end > total {
            return Err(RullamaError::Gguf(format!(
                "fetch_tensor_range({name}): range {byte_offset}..{end} extends past tensor end ({total})"
            )));
        }
        self.fetcher.fetch(start + byte_offset, byte_len).await
    }
}

fn align_up(x: u64, a: u64) -> u64 {
    if a <= 1 { x } else { x.div_ceil(a) * a }
}

// ---------- header parsing ----------

struct ParsedHeader {
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorDesc>,
    data_offset: usize,
    alignment: u64,
    version: u32,
}

fn parse_header(data: &[u8]) -> Result<ParsedHeader> {
    let mut c = Cursor { buf: data, pos: 0 };

    let magic = c.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(RullamaError::Gguf(format!(
            "bad magic 0x{magic:08x}, expected 0x{GGUF_MAGIC:08x} (GGUF)"
        )));
    }
    let version = c.read_u32()?;
    if version != SUPPORTED_VERSION {
        return Err(RullamaError::Gguf(format!(
            "unsupported GGUF version {version}, expected {SUPPORTED_VERSION}"
        )));
    }
    let tensor_count = c.read_u64()? as usize;
    let kv_count = c.read_u64()? as usize;

    // metadata
    let mut metadata: HashMap<String, GgufValue> = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = c.read_string()?;
        let vt = GgufValueType::from_u32(c.read_u32()?)?;
        let val = read_value(&mut c, vt)?;
        metadata.insert(key, val);
    }

    // tensor descriptors
    let mut tensors: Vec<TensorDesc> = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = c.read_string()?;
        let n_dims = c.read_u32()? as usize;
        if n_dims > 8 {
            return Err(RullamaError::Gguf(format!(
                "tensor {name} has {n_dims} dims (>8)"
            )));
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(c.read_u64()?);
        }
        let dtype = GgmlDtype::from_u32(c.read_u32()?)?;
        let offset = c.read_u64()?;
        tensors.push(TensorDesc {
            name,
            dims,
            dtype,
            offset,
        });
    }

    // tensor data starts after the descriptor section, aligned to general.alignment (default 32)
    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64().ok())
        .unwrap_or(DEFAULT_ALIGNMENT);
    let unaligned = c.pos as u64;
    let data_offset = align_up(unaligned, alignment) as usize;

    Ok(ParsedHeader {
        metadata,
        tensors,
        data_offset,
        alignment,
        version,
    })
}

// ---------- value decoding ----------

fn read_value(c: &mut Cursor<'_>, vt: GgufValueType) -> Result<GgufValue> {
    Ok(match vt {
        GgufValueType::U8 => GgufValue::U8(c.read_u8()?),
        GgufValueType::I8 => GgufValue::I8(c.read_u8()? as i8),
        GgufValueType::U16 => GgufValue::U16(c.read_u16()?),
        GgufValueType::I16 => GgufValue::I16(c.read_u16()? as i16),
        GgufValueType::U32 => GgufValue::U32(c.read_u32()?),
        GgufValueType::I32 => GgufValue::I32(c.read_u32()? as i32),
        GgufValueType::U64 => GgufValue::U64(c.read_u64()?),
        GgufValueType::I64 => GgufValue::I64(c.read_u64()? as i64),
        GgufValueType::F32 => GgufValue::F32(f32::from_bits(c.read_u32()?)),
        GgufValueType::F64 => GgufValue::F64(f64::from_bits(c.read_u64()?)),
        GgufValueType::Bool => GgufValue::Bool(c.read_u8()? != 0),
        GgufValueType::String => GgufValue::String(c.read_string()?),
        GgufValueType::Array => {
            let elem = GgufValueType::from_u32(c.read_u32()?)?;
            let n = c.read_u64()? as usize;
            read_array(c, elem, n)?
        }
    })
}

fn read_array(c: &mut Cursor<'_>, elem: GgufValueType, n: usize) -> Result<GgufValue> {
    Ok(match elem {
        GgufValueType::U8 => {
            let bytes = c.read_bytes(n)?.to_vec();
            GgufValue::ArrayU8(bytes)
        }
        GgufValueType::I8 => {
            let raw = c.read_bytes(n)?;
            GgufValue::ArrayI8(raw.iter().map(|&b| b as i8).collect())
        }
        GgufValueType::U16 => GgufValue::ArrayU16(c.read_u16_vec(n)?),
        GgufValueType::I16 => {
            GgufValue::ArrayI16(c.read_u16_vec(n)?.into_iter().map(|x| x as i16).collect())
        }
        GgufValueType::U32 => GgufValue::ArrayU32(c.read_u32_vec(n)?),
        GgufValueType::I32 => {
            GgufValue::ArrayI32(c.read_u32_vec(n)?.into_iter().map(|x| x as i32).collect())
        }
        GgufValueType::U64 => GgufValue::ArrayU64(c.read_u64_vec(n)?),
        GgufValueType::I64 => {
            GgufValue::ArrayI64(c.read_u64_vec(n)?.into_iter().map(|x| x as i64).collect())
        }
        GgufValueType::F32 => {
            GgufValue::ArrayF32(c.read_u32_vec(n)?.into_iter().map(f32::from_bits).collect())
        }
        GgufValueType::F64 => {
            GgufValue::ArrayF64(c.read_u64_vec(n)?.into_iter().map(f64::from_bits).collect())
        }
        GgufValueType::Bool => {
            let raw = c.read_bytes(n)?;
            GgufValue::ArrayBool(raw.iter().map(|&b| b != 0).collect())
        }
        GgufValueType::String => {
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(c.read_string()?);
            }
            GgufValue::ArrayString(out)
        }
        GgufValueType::Array => {
            return Err(RullamaError::Gguf(
                "nested arrays are not supported by GGUF v3".into(),
            ));
        }
    })
}

// ---------- raw byte cursor ----------

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn need(&self, n: usize) -> Result<()> {
        if self.pos + n > self.buf.len() {
            Err(RullamaError::Gguf(format!(
                "unexpected EOF: needed {n} bytes at {}, buffer len {}",
                self.pos,
                self.buf.len()
            )))
        } else {
            Ok(())
        }
    }
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.need(n)?;
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn read_u8(&mut self) -> Result<u8> {
        self.need(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn read_u16(&mut self) -> Result<u16> {
        self.need(2)?;
        let v = u16::from_le_bytes(self.buf[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }
    fn read_u32(&mut self) -> Result<u32> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn read_u64(&mut self) -> Result<u64> {
        self.need(8)?;
        let v = u64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }
    fn read_string(&mut self) -> Result<String> {
        let n = self.read_u64()? as usize;
        let bytes = self.read_bytes(n)?;
        // GGUF strings are UTF-8 by spec. We accept invalid bytes loosely so a single
        // mojibake token doesn't kill the whole load.
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
    fn read_u16_vec(&mut self, n: usize) -> Result<Vec<u16>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.read_u16()?);
        }
        Ok(out)
    }
    fn read_u32_vec(&mut self, n: usize) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.read_u32()?);
        }
        Ok(out)
    }
    fn read_u64_vec(&mut self, n: usize) -> Result<Vec<u64>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.read_u64()?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal synthetic GGUF: header, one u32 metadata kv, no tensors.
    fn synth() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        b.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count
        // kv: key "x" (u64 len + bytes), value_type=U32, value=42
        let key = b"x";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key);
        b.extend_from_slice(&(GgufValueType::U32 as u32).to_le_bytes());
        b.extend_from_slice(&42u32.to_le_bytes());
        // pad to 32-byte alignment for the (empty) data section
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b
    }

    #[test]
    fn parses_minimal_gguf() {
        let bytes = synth();
        let r = GgufReader::new(bytes).expect("parse");
        assert_eq!(r.version(), 3);
        assert_eq!(r.tensors().len(), 0);
        assert_eq!(r.get("x").unwrap().as_u32().unwrap(), 42);
        assert!(r.is_in_memory());
    }

    #[test]
    fn streaming_reader_matches_in_memory() {
        let bytes = synth();
        let in_mem = GgufReader::new(bytes.clone()).expect("parse");

        let fetcher: Arc<dyn TensorFetcher> = Arc::new(InMemoryFetcher::new(bytes));
        let streamed = pollster::block_on(GgufReader::new_streaming(fetcher)).expect("stream");

        assert_eq!(streamed.version(), in_mem.version());
        assert_eq!(streamed.tensors().len(), in_mem.tensors().len());
        assert_eq!(
            streamed.get("x").unwrap().as_u32().unwrap(),
            in_mem.get("x").unwrap().as_u32().unwrap()
        );
        assert!(!streamed.is_in_memory());
        assert!(streamed.tensor_bytes("anything").is_err()); // streaming → must use async
    }
}
