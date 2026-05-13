//! Quick token-id → string lookup using the vocab embedded in the GGUF. No tokenizer
//! mechanics — just an array indexing.

use std::env;
use std::fs;
use std::process::ExitCode;

use rullama::gguf::GgufReader;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => { eprintln!("usage: decode_ids <gguf> <id1> <id2> ..."); return ExitCode::from(2); }
    };
    let bytes = fs::read(&path).expect("read");
    let r = GgufReader::new(bytes).expect("parse");
    let tokens = r.get("tokenizer.ggml.tokens").expect("vocab").as_string_array().expect("strs");
    for arg in args {
        let id: usize = arg.parse().expect("id");
        if id < tokens.len() {
            println!("{:>7}  {:?}", id, tokens[id]);
        } else {
            println!("{id} OUT OF RANGE");
        }
    }
    ExitCode::SUCCESS
}
