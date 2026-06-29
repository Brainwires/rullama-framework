//! Minimal safetensors blob reader for Ollama image-model tensors.
//!
//! Each Ollama image blob is a standalone safetensors v1 file:
//! ```text
//! [8 bytes: header_size (u64 LE)] [header_size bytes: JSON] [tensor data]
//! ```
//! The JSON header maps tensor name → `{dtype, shape, data_offsets:[start,end]}`
//! (offsets are relative to the start of the data region), plus an optional
//! `__metadata__` string map carrying grouped-quant info
//! (`quant_type`, `group_size`).
//!
//! We hand-parse rather than lean on the `safetensors` crate so we can carry
//! the F8 / quant-metadata semantics the crate doesn't model, and so a tensor's
//! bytes can be sliced without materializing the rest of the blob. A blob holds
//! a single logical tensor (or a packed weight + its `.scale`/`.bias`
//! companions), so it's small enough to hold whole while we extract and upload.
//!
//! Reference: Ollama `x/imagegen/docs/blob-format.md`.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::error::{Result, RullamaError};
use crate::imagegen::dtype::StDtype;

/// One tensor entry from the header.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub dtype: StDtype,
    pub shape: Vec<usize>,
    /// `[start, end)` byte range within the data region. `u64` (not `usize`)
    /// because multi-GB shards carry offsets past `u32::MAX`, which would
    /// overflow `usize` on wasm32 during header deserialization.
    pub data_offsets: (u64, u64),
}

impl TensorEntry {
    /// Number of elements (product of shape; 1 for a scalar).
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }
}

/// Header-only view of a blob: tensor table + `__metadata__` + the byte offset
/// where the data region starts, parsed from just the leading
/// `8 + header_size` bytes. With `data_start` + a tensor's `data_offsets`, a
/// caller can range-read one tensor without materializing the rest of the
/// (possibly multi-GB) blob.
#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    pub header_size: usize,
    /// Absolute byte offset of the data region (`8 + header_size`).
    pub data_start: usize,
    pub tensors: BTreeMap<String, TensorEntry>,
    pub metadata: BTreeMap<String, String>,
}

impl SafetensorsHeader {
    /// Absolute `[start, end)` byte range of a tensor within the blob/file.
    pub fn tensor_range(&self, name: &str) -> Option<(u64, u64)> {
        self.tensors.get(name).map(|e| {
            (
                self.data_start as u64 + e.data_offsets.0,
                self.data_start as u64 + e.data_offsets.1,
            )
        })
    }
}

/// Parse only the header from a blob prefix (must contain at least
/// `8 + header_size` bytes). Skips data-region bounds checks.
pub fn read_header(prefix: &[u8]) -> Result<SafetensorsHeader> {
    if prefix.len() < 8 {
        return Err(RullamaError::Image("blob prefix < 8 bytes".into()));
    }
    let header_size = u64::from_le_bytes(prefix[0..8].try_into().expect("8 bytes")) as usize;
    let end = 8usize
        .checked_add(header_size)
        .ok_or_else(|| RullamaError::Image("header size overflow".into()))?;
    if end > prefix.len() {
        return Err(RullamaError::Image(format!(
            "prefix too short: need {end} bytes for header, have {}",
            prefix.len()
        )));
    }
    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&prefix[8..end])
        .map_err(|e| RullamaError::Image(format!("safetensors header JSON: {e}")))?;
    let mut tensors = BTreeMap::new();
    let mut metadata = BTreeMap::new();
    for (name, val) in raw {
        if name == "__metadata__" {
            if let serde_json::Value::Object(map) = val {
                for (k, v) in map {
                    if let serde_json::Value::String(s) = v {
                        metadata.insert(k, s);
                    }
                }
            }
            continue;
        }
        let e: RawEntry = serde_json::from_value(val)
            .map_err(|err| RullamaError::Image(format!("tensor {name:?}: {err}")))?;
        tensors.insert(
            name,
            TensorEntry {
                dtype: StDtype::parse(&e.dtype)?,
                shape: e.shape,
                data_offsets: (e.data_offsets[0], e.data_offsets[1]),
            },
        );
    }
    Ok(SafetensorsHeader {
        header_size,
        data_start: end,
        tensors,
        metadata,
    })
}

/// A parsed safetensors blob: header metadata + tensor table + the data region
/// offset. Owns the blob bytes so tensor slices stay valid.
pub struct SafetensorsBlob {
    bytes: Vec<u8>,
    data_start: usize,
    tensors: BTreeMap<String, TensorEntry>,
    metadata: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct RawEntry {
    dtype: String,
    #[serde(default)]
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

impl SafetensorsBlob {
    /// Parse a complete blob.
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(RullamaError::Image("safetensors blob < 8 bytes".into()));
        }
        let header_size = u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")) as usize;
        let header_end = 8usize
            .checked_add(header_size)
            .ok_or_else(|| RullamaError::Image("header size overflow".into()))?;
        if header_end > bytes.len() {
            return Err(RullamaError::Image(format!(
                "header_size {header_size} exceeds blob ({} bytes)",
                bytes.len()
            )));
        }
        let data_start = header_end;
        let data_len = bytes.len() - data_start;

        // The header is a JSON object: tensor entries + optional `__metadata__`.
        let raw: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..header_end])
                .map_err(|e| RullamaError::Image(format!("safetensors header JSON: {e}")))?;

        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, val) in raw {
            if name == "__metadata__" {
                if let serde_json::Value::Object(map) = val {
                    for (k, v) in map {
                        if let serde_json::Value::String(s) = v {
                            metadata.insert(k, s);
                        }
                    }
                }
                continue;
            }
            let e: RawEntry = serde_json::from_value(val)
                .map_err(|err| RullamaError::Image(format!("tensor {name:?}: {err}")))?;
            let (start, end) = (e.data_offsets[0], e.data_offsets[1]);
            if start > end || end > data_len as u64 {
                return Err(RullamaError::Image(format!(
                    "tensor {name:?} offsets [{start},{end}) out of data region ({data_len})"
                )));
            }
            let dtype = StDtype::parse(&e.dtype)?;
            let entry = TensorEntry {
                dtype,
                shape: e.shape,
                data_offsets: (start, end),
            };
            // Byte length must match dtype * elem_count.
            let expect = (entry.elem_count() * dtype.elem_size()) as u64;
            if end - start != expect {
                return Err(RullamaError::Image(format!(
                    "tensor {name:?}: byte span {} != {expect} ({:?} × {} elems)",
                    end - start,
                    dtype,
                    entry.elem_count()
                )));
            }
            tensors.insert(name, entry);
        }

        Ok(Self {
            bytes,
            data_start,
            tensors,
            metadata,
        })
    }

    /// Names of all tensors in the blob.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.get(name)
    }

    /// `__metadata__` map (quant_type / group_size live here when present).
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// `quant_type` from `__metadata__`, if this blob is quantized.
    pub fn quant_type(&self) -> Option<&str> {
        self.metadata.get("quant_type").map(String::as_str)
    }

    /// Raw bytes for a tensor (a slice into the data region).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let e = self
            .get(name)
            .ok_or_else(|| RullamaError::Image(format!("tensor {name:?} not in blob")))?;
        let (s, end) = e.data_offsets;
        Ok(&self.bytes[self.data_start + s as usize..self.data_start + end as usize])
    }

    /// Dequantize a *float* tensor to f32. Errors for integer/packed-quant
    /// dtypes — those need the grouped-quant reconstruction path.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let e = self
            .get(name)
            .ok_or_else(|| RullamaError::Image(format!("tensor {name:?} not in blob")))?;
        e.dtype.dequant_to_f32(self.tensor_bytes(name)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    /// Build a one-tensor safetensors blob in memory.
    fn make_blob(
        name: &str,
        dtype: &str,
        shape: &[usize],
        data: &[u8],
        meta: Option<&str>,
    ) -> Vec<u8> {
        let mut header = format!(
            "{{\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[0,{}]}}",
            data.len()
        );
        if let Some(m) = meta {
            header.push_str(m);
        }
        header.push('}');
        let hbytes = header.into_bytes();
        let mut out = Vec::new();
        out.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&hbytes);
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn parse_and_dequant_bf16_tensor() {
        let vals = [1.0f32, 2.0, -3.0, 0.5];
        let mut data = Vec::new();
        for v in vals {
            data.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
        }
        let blob = make_blob("transformer.weight", "BF16", &[2, 2], &data, None);
        let st = SafetensorsBlob::parse(blob).unwrap();
        assert_eq!(st.names().collect::<Vec<_>>(), ["transformer.weight"]);
        let got = st.tensor_f32("transformer.weight").unwrap();
        assert_eq!(got.len(), 4);
        for (g, v) in got.iter().zip(vals) {
            assert_eq!(*g, bf16::from_f32(v).to_f32());
        }
    }

    #[test]
    fn parses_quant_metadata() {
        let data = [0u8; 8]; // 8 × U8
        let meta = ",\"__metadata__\":{\"quant_type\":\"int4\",\"group_size\":\"32\"}";
        let blob = make_blob("w.packed", "U8", &[8], &data, Some(meta));
        let st = SafetensorsBlob::parse(blob).unwrap();
        assert_eq!(st.quant_type(), Some("int4"));
        assert_eq!(
            st.metadata().get("group_size").map(String::as_str),
            Some("32")
        );
        // U8 packed weight can't be float-dequantized directly.
        assert!(st.tensor_f32("w.packed").is_err());
    }

    #[test]
    fn rejects_offsets_past_data() {
        // header claims 16 bytes of data but blob only carries 4
        let header = "{\"x\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}";
        let hb = header.as_bytes();
        let mut blob = Vec::new();
        blob.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        blob.extend_from_slice(hb);
        blob.extend_from_slice(&[0u8; 4]);
        assert!(SafetensorsBlob::parse(blob).is_err());
    }

    #[test]
    fn rejects_byte_span_mismatch() {
        // F32 × 4 elems = 16 bytes, but we declare shape [2] (8 bytes) over a 16-byte span
        let header = "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,16]}}";
        let hb = header.as_bytes();
        let mut blob = Vec::new();
        blob.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        blob.extend_from_slice(hb);
        blob.extend_from_slice(&[0u8; 16]);
        assert!(SafetensorsBlob::parse(blob).is_err());
    }
}
