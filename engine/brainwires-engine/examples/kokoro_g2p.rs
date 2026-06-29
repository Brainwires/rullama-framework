//! G2P validation: run the lexicon G2P and diff against a misaki reference corpus.
//!
//!   cargo run --release --example kokoro_g2p -- \
//!       ~/.cache/kokoro/us_gold.json ~/.cache/kokoro/g2p_corpus.json

use rullama::reference::kokoro::g2p::{Lexicon, g2p};
use std::fs;

fn main() {
    let mut args = std::env::args().skip(1);
    let lex_path = args
        .next()
        .expect("usage: kokoro_g2p <us_gold.json> <corpus.json>");
    let corpus_path = args
        .next()
        .expect("usage: kokoro_g2p <us_gold.json> <corpus.json>");

    // optional 3rd arg: us_silver.json fallback
    let silver = std::env::args()
        .nth(3)
        .map(|p| fs::read(p).unwrap())
        .unwrap_or_default();
    let lex = Lexicon::load(&fs::read(&lex_path).unwrap(), &silver);
    println!("lexicon: {} entries", lex.len());

    let corpus: serde_json::Value =
        serde_json::from_slice(&fs::read(&corpus_path).unwrap()).unwrap();
    let mut exact = 0;
    let total = corpus.as_array().unwrap().len();
    for row in corpus.as_array().unwrap() {
        let text = row["text"].as_str().unwrap();
        let want = row["phonemes"].as_str().unwrap();
        let (got, oov) = g2p(text, &lex);
        let ok = got == want;
        if ok {
            exact += 1;
        }
        println!("{} {:?}", if ok { "OK  " } else { "DIFF" }, text);
        if !ok {
            println!("    misaki: {want}");
            println!("    ours:   {got}");
            if !oov.is_empty() {
                println!("    OOV:    {oov:?}");
            }
        }
    }
    println!("\nexact phrase match: {exact}/{total}");
}
