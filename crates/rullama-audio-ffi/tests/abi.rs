//! C ABI behaviour that needs neither a model nor an audio device.
//!
//! Nothing here opens the microphone: only calls that are rejected before any
//! device is touched (wrong state, no model, bad path) are made.

use std::ffi::CStr;

use rullama_audio_ffi::*;

fn last_error() -> String {
    let mut buf = [0u8; 512];
    let n = unsafe { rla_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(n >= 0);
    String::from_utf8(buf[..n as usize].to_vec()).unwrap()
}

#[test]
fn idle_abi_contract() {
    let v = unsafe { CStr::from_ptr(rla_version()) }.to_str().unwrap();
    assert!(v.starts_with("rullama-audio-ffi "), "{v}");

    assert_eq!(rla_state(), 0);
    assert_eq!(rla_stt_loaded(), 0);
    assert_eq!(rla_level(), 0.0);
    assert_eq!(rla_captured_ms(), 0);
    assert_eq!(rla_playing(), 0);

    assert_eq!(rla_capture_stop(), -2);
    assert!(last_error().contains("not capturing"));
    let out = c"/tmp/rla-should-not-exist.wav";
    assert_eq!(unsafe { rla_capture_stop_wav(out.as_ptr()) }, -2);

    let mut buf = [0u8; 16];
    assert_eq!(
        unsafe { rla_take_transcript(buf.as_mut_ptr(), buf.len()) },
        -2
    );
    assert_eq!(rla_cancel(), 0);
    assert_eq!(rla_state(), 0);
    assert_eq!(rla_play_stop(), 0);

    // No model: -3 (the WAV is never read).
    let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/phrase.wav\0");
    let n = unsafe { rla_transcribe_wav(fixture.as_ptr().cast(), buf.as_mut_ptr(), buf.len()) };
    assert_eq!(n, -3);
    assert!(last_error().contains("no speech model"));

    // Missing file for playback is an error, raised before any device access.
    let missing = c"/nonexistent/rla.wav";
    assert_eq!(unsafe { rla_play_wav(missing.as_ptr()) }, -1);
    assert!(last_error().contains("/nonexistent/rla.wav"));
    assert_eq!(rla_playing(), 0);

    // rla_last_error truncates and reports the full length.
    let full = last_error();
    let mut small = [0xAAu8; 5];
    let n = unsafe { rla_last_error(small.as_mut_ptr(), small.len()) };
    assert_eq!(n as usize, full.len());
    assert_eq!(&small[..4], &full.as_bytes()[..4]);
    assert_eq!(small[4], 0);

    // Shutdown is safe when idle and repeatable, and leaves the library usable.
    for _ in 0..3 {
        assert_eq!(rla_shutdown(), 0);
        assert_eq!(rla_state(), 0);
        assert_eq!(rla_stt_loaded(), 0);
        assert_eq!(rla_playing(), 0);
    }
    assert_eq!(rla_capture_stop(), -2);
}

/// The header's prototypes, as `name -> (return type, parameter types)`, with
/// the Rust signature each one must have. The fn-pointer coercions are checked
/// by the compiler; the strings are checked against `include/rullama_audio.h`.
#[test]
fn header_matches_rust_signatures() {
    use std::ffi::c_char;

    // Rust side: fails to compile if any exported signature drifts.
    let _: extern "C" fn() -> *const c_char = rla_version;
    let _: unsafe extern "C" fn(*mut u8, usize) -> i32 = rla_last_error;
    let _: unsafe extern "C" fn(*const c_char) -> i32 = rla_stt_load;
    let _: extern "C" fn() -> i32 = rla_stt_loaded;
    let _: extern "C" fn() -> i32 = rla_capture_start;
    let _: extern "C" fn() -> i32 = rla_capture_stop;
    let _: unsafe extern "C" fn(*const c_char) -> i32 = rla_capture_stop_wav;
    let _: extern "C" fn() -> i32 = rla_cancel;
    let _: extern "C" fn() -> i32 = rla_state;
    let _: extern "C" fn() -> f32 = rla_level;
    let _: extern "C" fn() -> u32 = rla_captured_ms;
    let _: unsafe extern "C" fn(*mut u8, usize) -> i32 = rla_take_transcript;
    let _: unsafe extern "C" fn(*const c_char, *mut u8, usize) -> i32 = rla_transcribe_wav;
    let _: unsafe extern "C" fn(*const c_char) -> i32 = rla_play_wav;
    let _: extern "C" fn() -> i32 = rla_play_stop;
    let _: extern "C" fn() -> i32 = rla_playing;
    let _: extern "C" fn() -> i32 = rla_shutdown;

    // C side, with the C types that correspond to the Rust ones above
    // (int32_t = i32, uint32_t = u32, float = f32, size_t = usize,
    // uint8_t * = *mut u8, const char * = *const c_char).
    let expected: &[(&str, &str, &[&str])] = &[
        ("rla_version", "const char *", &[]),
        ("rla_last_error", "int32_t", &["uint8_t *", "size_t"]),
        ("rla_stt_load", "int32_t", &["const char *"]),
        ("rla_stt_loaded", "int32_t", &[]),
        ("rla_capture_start", "int32_t", &[]),
        ("rla_capture_stop", "int32_t", &[]),
        ("rla_capture_stop_wav", "int32_t", &["const char *"]),
        ("rla_cancel", "int32_t", &[]),
        ("rla_state", "int32_t", &[]),
        ("rla_level", "float", &[]),
        ("rla_captured_ms", "uint32_t", &[]),
        ("rla_take_transcript", "int32_t", &["uint8_t *", "size_t"]),
        (
            "rla_transcribe_wav",
            "int32_t",
            &["const char *", "uint8_t *", "size_t"],
        ),
        ("rla_play_wav", "int32_t", &["const char *"]),
        ("rla_play_stop", "int32_t", &[]),
        ("rla_playing", "int32_t", &[]),
        ("rla_shutdown", "int32_t", &[]),
    ];

    let header = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/include/rullama_audio.h"
    ))
    .unwrap();
    let parsed = parse_prototypes(&header);
    let names: Vec<&str> = parsed.iter().map(|p| p.0.as_str()).collect();
    let expected_names: Vec<&str> = expected.iter().map(|e| e.0).collect();
    assert_eq!(
        names, expected_names,
        "header declares a different set of functions"
    );
    for ((name, ret, params), (ename, eret, eparams)) in parsed.iter().zip(expected) {
        assert_eq!(
            (name.as_str(), ret.as_str()),
            (*ename, *eret),
            "return type of {name}"
        );
        assert_eq!(params, eparams, "parameter types of {name}");
    }

    // The #defines match the Rust constants.
    let defines = [
        ("RLA_OK", rullama_audio_ffi::error::OK),
        ("RLA_ERROR", rullama_audio_ffi::error::ERROR),
        ("RLA_WRONG_STATE", rullama_audio_ffi::error::WRONG_STATE),
        ("RLA_NO_MODEL", rullama_audio_ffi::error::NO_MODEL),
        ("RLA_NO_DEVICE", rullama_audio_ffi::error::NO_DEVICE),
        ("RLA_STATE_IDLE", rullama_audio_ffi::engine::STATE_IDLE),
        (
            "RLA_STATE_CAPTURING",
            rullama_audio_ffi::engine::STATE_CAPTURING,
        ),
        (
            "RLA_STATE_TRANSCRIBING",
            rullama_audio_ffi::engine::STATE_TRANSCRIBING,
        ),
        ("RLA_STATE_DONE", rullama_audio_ffi::engine::STATE_DONE),
        ("RLA_STATE_ERROR", rullama_audio_ffi::engine::STATE_ERROR),
    ];
    for (name, value) in defines {
        let line = header
            .lines()
            .find(|l| l.split_whitespace().nth(1) == Some(name) && l.starts_with("#define"))
            .unwrap_or_else(|| panic!("missing #define {name}"));
        let text: String = line
            .split_whitespace()
            .nth(2)
            .unwrap()
            .chars()
            .filter(|c| *c != '(' && *c != ')')
            .collect();
        assert_eq!(text.parse::<i32>().unwrap(), value, "#define {name}");
    }
}

/// Declarations of the form `<type> rla_x(<params>);`, comments stripped,
/// parameter names dropped, `*` spacing normalised to `type *`.
fn parse_prototypes(header: &str) -> Vec<(String, String, Vec<String>)> {
    let mut code = String::new();
    let mut rest = header;
    while let Some(start) = rest.find("/*") {
        code.push_str(&rest[..start]);
        let end = rest[start..].find("*/").expect("unterminated comment") + start + 2;
        rest = &rest[end..];
    }
    code.push_str(rest);

    let norm_type = |t: &str| {
        let t = t.replace('*', " * ");
        let words: Vec<&str> = t.split_whitespace().collect();
        words.join(" ")
    };
    // Each declaration ends with `;` and starts on its own line, after any
    // preprocessor lines or earlier declarations.
    code.split(';')
        .map(|chunk| chunk.trim().rsplit('\n').next().unwrap_or("").trim())
        .filter(|d| d.contains("rla_") && d.ends_with(')') && !d.starts_with('#'))
        .map(|decl| {
            let open = decl.find('(').unwrap();
            let head = &decl[..open];
            let name_at = head
                .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
                .unwrap()
                + 1;
            let name = head[name_at..].to_string();
            let ret = norm_type(&head[..name_at]);
            let inner = &decl[open + 1..decl.len() - 1];
            let params = if inner.trim() == "void" {
                Vec::new()
            } else {
                inner
                    .split(',')
                    .map(|p| {
                        // Drop the parameter name (the last identifier).
                        let p = p.trim();
                        let cut = p
                            .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
                            .map_or(0, |i| i + 1);
                        norm_type(&p[..cut])
                    })
                    .collect()
            };
            (name, ret, params)
        })
        .collect()
}
