//! GGUF v3 parser.
//!
//! Browser-friendly: takes `&[u8]` (a `Uint8Array` slice on wasm32), no mmap, no I/O.
//! Hand-rolled rather than depending on a crate so we own the wasm story end-to-end and
//! the dep tree stays small.
//!
//! Spec reference: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

mod dtype;
pub mod fetcher;
mod reader;
mod value;

pub mod quant;
pub mod tensor;

pub use dtype::GgmlDtype;
#[cfg(not(target_arch = "wasm32"))]
pub use fetcher::FileFetcher;
#[cfg(target_arch = "wasm32")]
pub use fetcher::HttpRangeFetcher;
#[cfg(target_arch = "wasm32")]
pub use fetcher::OpfsFetcher;
pub use fetcher::{InMemoryFetcher, TensorFetcher};
pub use reader::{GgufReader, TensorDesc};
pub use tensor::{
    dequant_expert_slice_to_f32, dequant_expert_slice_to_f32_async, dequant_row_to_f32,
    dequant_row_to_f32_async, dequant_tensor_to_f16, dequant_tensor_to_f16_async,
    dequant_tensor_to_f32, dequant_tensor_to_f32_async,
};
pub use value::{GgufValue, GgufValueType};
