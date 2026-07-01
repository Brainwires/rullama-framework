//! Quick: tokenize a string with our (Ollama-matching) BPE and print the IDs.
use rullama_engine::gguf::GgufReader;
use rullama_engine::tokenizer::BpeTokenizer;
use std::env;
use std::fs;

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: encode_lookup <gguf> <text>");
    let text: String = args.collect::<Vec<_>>().join(" ");
    let bytes = fs::read(&path).expect("read");
    let r = GgufReader::new(bytes).expect("parse");
    let tok = BpeTokenizer::from_gguf(&r).expect("tokenizer");
    let ids = tok.encode(&text);
    println!("input:  {:?}", text);
    println!("ids:    {:?}", ids);
    for id in &ids {
        let s = tok.id_to_str(*id).unwrap_or("?");
        println!("  {id:>6}  {s:?}");
    }
}
