//! Smoke-test the BPE tokenizer against Ollama's golden vectors from
//! `model/models/gemma4/tokenizer_compare_test.go`.

use std::env;
use std::fs;
use std::process::ExitCode;

use brainwires_engine::gguf::GgufReader;
use brainwires_engine::tokenizer::BpeTokenizer;

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: encode_check <gguf>");
            return ExitCode::from(2);
        }
    };
    let bytes = fs::read(&path).expect("read");
    let r = GgufReader::new(bytes).expect("parse");
    let tok = match BpeTokenizer::from_gguf(&r) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer build error: {e}");
            return ExitCode::from(1);
        }
    };
    println!("vocab_size = {}", tok.vocab_size());

    let cases: &[(&str, &[u32])] = &[
        ("Hello, world!", &[9259, 236764, 1902, 236888]),
        (
            "<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n",
            &[
                105, 2364, 107, 3689, 563, 236743, 236778, 236862, 236778, 236881, 106, 107, 105,
                4368, 107,
            ],
        ),
    ];

    let mut all_pass = true;
    for (input, expected) in cases {
        let got = tok.encode(input);
        let ok = got == *expected;
        if !ok {
            all_pass = false;
        }
        println!(
            "{} {:?}\n  got      = {:?}\n  expected = {:?}",
            if ok { "PASS" } else { "FAIL" },
            input,
            got,
            expected
        );
    }
    if all_pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
