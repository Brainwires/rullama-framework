//! Gemma 4 special tokens.
//!
//! Source: `model/renderers/gemma4.go` and `convert/convert_gemma4.go` in the
//! Ollama reference impl at /Users/nightness/Source/ollama.

pub const BOS: &str = "<bos>";
pub const TURN_OPEN: &str = "<|turn>";
pub const TURN_CLOSE: &str = "<turn|>";
pub const EOT_ID: u32 = 106;
