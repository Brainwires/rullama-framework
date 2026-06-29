/// Streaming JSONL reader.
pub mod reader;
/// Buffered JSONL writer.
pub mod writer;

pub use reader::{JsonlReader, read_jsonl, read_jsonl_preferences};
pub use writer::{JsonlWriter, write_jsonl, write_jsonl_preferences};
