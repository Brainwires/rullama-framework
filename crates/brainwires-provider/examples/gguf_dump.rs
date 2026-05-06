//! Diagnostic: dump every GGUF metadata key + tensor name in a model.
//!
//! Used to verify which metadata schema an Ollama publication actually
//! uses (vs the llama.cpp-style names quantized_gemma4 guesses for
//! AltUp / PLE / Laurel weights).

use anyhow::Result;
use candle_core::quantized::gguf_file;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gguf_dump <path-to-.gguf>");
    let mut f = std::fs::File::open(&path)?;
    let ct = gguf_file::Content::read(&mut f)?;

    println!("=== METADATA ({} keys) ===", ct.metadata.len());
    let mut keys: Vec<&String> = ct.metadata.keys().collect();
    keys.sort();
    for k in keys {
        let v = &ct.metadata[k];
        let s = format!("{v:?}");
        let preview = if s.len() > 240 {
            format!("{}…({} chars)", &s[..240], s.len())
        } else {
            s
        };
        println!("  {k} = {preview}");
    }

    println!("\n=== TENSORS ({} entries) ===", ct.tensor_infos.len());
    let mut names: Vec<&String> = ct.tensor_infos.keys().collect();
    names.sort();
    for n in names {
        let ti = &ct.tensor_infos[n];
        println!(
            "  {n}  dtype={:?}  shape={:?}",
            ti.ggml_dtype,
            ti.shape.dims()
        );
    }
    Ok(())
}
