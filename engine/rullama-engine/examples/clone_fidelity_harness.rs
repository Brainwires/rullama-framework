//! A0 — clone-fidelity calibration harness.
//!
//! Tests Phase A's premise (clean abundant reference → faithful clone) by removing the two
//! confounds in a real user recording (mic noise, OOD speaker): use Kokoro `af_heart` — a
//! clean, consistent preset — as the "speaker," clone IT through the StyleTTS2 pipeline, and
//! measure how well the clone reproduces af_heart's identity.
//!
//! Pipeline: Kokoro af_heart → N varied clips → StyleTTS2 encode each → trimmed-mean → cloned
//! voice → synth held-out sentences → re-encode → speaker-similarity vs af_heart's own outputs.
//!
//! Identity metric = cosine of the **acoustic half** (first 128 dims = timbre) of the StyleTTS2
//! style vector. The prosodic half is excluded — it's prosody, not identity. Calibrated by:
//!   CEILING  — af_heart-output vs af_heart-output (same speaker, different text)
//!   FLOOR    — af_heart-output vs a different Kokoro speaker's output
//! A clone is "good" when its similarity approaches CEILING and sits well above FLOOR.
//!
//!   cargo run -p rullama-engine --release --example clone_fidelity_harness -- \
//!       ~/.cache/kokoro/kokoro-82m-f32.gguf ~/.cache/styletts2/styletts2-libritts-f32.gguf \
//!       ~/.cache/kokoro/us_gold.json [~/.cache/kokoro/us_silver.json] [different_voice=am_michael]

use rullama_engine::backend::{Pipelines, WgpuCtx};
use rullama_engine::gguf::GgufReader;
use rullama_engine::reference::kokoro::KokoroModel;
use rullama_engine::reference::kokoro::g2p::{Lexicon, g2p};
use rullama_engine::reference::styletts2::StyleTtsModel;
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

const SR: f32 = 24000.0;
const STYLE_DIM: usize = 128; // acoustic half width
// GPU-gentle pacing: this runs on an integrated GPU where sustained saturation trips the macOS
// watchdog. Sleep between every GPU op so the GPU drains, and cap the corpus.
// The real throttle is INTERNAL: ST2_GPU_THROTTLE_MS makes every StyleTTS2 GPU stage drain+sleep,
// so the GPU yields mid-synth. These coarse gaps are just a small extra cushion between ops.
const THROTTLE_MS: u64 = 300; // after each Kokoro synth / style encode
const COOL_MS: u64 = 500; // after a full StyleTTS2 synth (which already yielded internally)
const MAX_REF: usize = 16; // reference clips actually used (≈1.5-2 min of audio)
fn breathe() {
    std::thread::sleep(std::time::Duration::from_millis(THROTTLE_MS));
}
fn cool() {
    std::thread::sleep(std::time::Duration::from_millis(COOL_MS));
}

// Varied reference corpus (builds the cloned voice). Distinct sentences = real averaging.
const REF_CORPUS: &[&str] = &[
    "The morning light slipped quietly across the kitchen floor.",
    "She wondered whether the train would arrive on time today.",
    "A gentle breeze carried the scent of rain through the open window.",
    "Numbers and letters danced together on the crowded whiteboard.",
    "He folded the letter carefully and placed it in his coat pocket.",
    "The old bridge creaked under the weight of the passing cart.",
    "Bright orange leaves gathered along the edge of the path.",
    "We talked for hours about books, music, and distant cities.",
    "The recipe called for two eggs, a cup of flour, and patience.",
    "Thunder rolled somewhere beyond the line of dark green hills.",
    "Her laughter echoed down the long marble hallway.",
    "The children built a castle of sand near the rising tide.",
    "A single candle flickered against the cold stone wall.",
    "He measured the board twice before making the first cut.",
    "The market was alive with color, noise, and the smell of spices.",
    "Snow fell softly, covering the rooftops in a quiet blanket.",
    "She practiced the difficult passage until her fingers ached.",
    "The lighthouse blinked steadily through the gathering fog.",
    "They planted rows of tomatoes along the southern fence.",
    "An owl called twice from the shadows of the ancient oak.",
    "The professor scribbled an equation and underlined it twice.",
    "Waves crashed against the rocks in a slow, steady rhythm.",
    "He whispered the answer so only she could hear it.",
    "The garden bloomed with roses, lavender, and wild daisies.",
    "A distant violin played a melody both sad and sweet.",
    "The map was old, its corners worn and its ink faded.",
    "She poured the coffee and watched the steam curl upward.",
    "The engine sputtered once, then roared confidently to life.",
    "Stars scattered across the sky like grains of silver sand.",
    "He counted the stairs as he climbed toward the attic door.",
    "The puppy chased its tail in dizzy, happy circles.",
    "Rain tapped against the glass like a hundred tiny drummers.",
    "They signed the papers and shook hands with quiet relief.",
    "A warm loaf of bread cooled on the windowsill at noon.",
    "The clock tower struck twelve as the crowd fell silent.",
    "She closed her eyes and listened to the river running by.",
];

// Held-out sentences (NOT used to build any voice) — the synthesis test set.
const HELD_OUT: &[&str] = &[
    "Honestly, this is the part where everything finally comes together.",
    "Could you hand me that blue notebook on the table, please?",
    "I never expected the weekend to turn out quite like this.",
];

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-9)
}

/// acoustic (timbre) half of a 256-d StyleTTS2 style vector.
fn acoustic(v: &[f32]) -> &[f32] {
    &v[..STYLE_DIM]
}

/// trimmed mean across style vectors: drop the 20% farthest from the median-ish centroid.
fn trimmed_mean(vs: &[Vec<f32>]) -> Vec<f32> {
    if vs.len() <= 2 {
        return mean(vs);
    }
    let c = mean(vs);
    let mut scored: Vec<(f32, &Vec<f32>)> = vs
        .iter()
        .map(|v| (1.0 - cosine(acoustic(v), acoustic(&c)), v))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let keep = ((vs.len() as f32) * 0.8).ceil() as usize;
    mean(
        &scored[..keep]
            .iter()
            .map(|(_, v)| (*v).clone())
            .collect::<Vec<_>>(),
    )
}

fn mean(vs: &[Vec<f32>]) -> Vec<f32> {
    let d = vs[0].len();
    let mut m = vec![0f32; d];
    for v in vs {
        for k in 0..d {
            m[k] += v[k];
        }
    }
    for x in m.iter_mut() {
        *x /= vs.len() as f32;
    }
    m
}

fn main() {
    let mut a = std::env::args().skip(1);
    let kokoro_gguf = a.next().expect(
        "usage: <kokoro.gguf> <styletts2.gguf> <us_gold.json> [us_silver.json] [diff_voice]",
    );
    let st_gguf = a.next().expect("styletts2 gguf");
    let gold = fs::read(a.next().expect("us_gold.json")).unwrap();
    let silver = a.next().map(|p| fs::read(p).unwrap()).unwrap_or_default();
    let diff_voice = a.next().unwrap_or_else(|| "am_michael".into());

    // engines
    let kokoro = KokoroModel::new(Arc::new(
        GgufReader::new(fs::read(&kokoro_gguf).unwrap()).unwrap(),
    ))
    .unwrap();
    let lex = Lexicon::load(&gold, &silver);
    let st = StyleTtsModel::load(&GgufReader::new(fs::read(&st_gguf).unwrap()).unwrap()).unwrap();
    let ctx = pollster::block_on(WgpuCtx::new()).expect("wgpu");
    let pipes = Pipelines::new(&ctx.device);
    let mut wc = rullama_engine::reference::styletts2::gpu::GpuWeightCache::new();

    // StyleTTS2 phoneme vocab (BOS=0 prefix; drop OOV)
    let vocab_txt = fs::read_to_string(format!(
        "{}/src/reference/styletts2/vocab.txt",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let vocab: HashMap<char, i64> = vocab_txt
        .chars()
        .enumerate()
        .map(|(i, c)| (c, i as i64))
        .collect();
    let to_ids = |text: &str| -> Vec<i64> {
        let (ps, _) = g2p(text, &lex);
        let mut ids = vec![0i64];
        ids.extend(ps.chars().filter_map(|c| vocab.get(&c).copied()));
        ids
    };
    let ksynth = |text: &str, voice: &str| -> Vec<f32> {
        let r = pollster::block_on(kokoro.synthesize_text_gpu(&ctx, &pipes, text, voice, &lex)).0;
        breathe();
        r
    };

    macro_rules! enc {
        ($pcm:expr) => {{
            let v = pollster::block_on(st.encode_voice_gpu(&ctx, &pipes, &mut wc, $pcm, None));
            breathe();
            v
        }};
    }
    // synth held-out set with `voice` (diffusion OFF = deterministic, identity-focused) and
    // report mean acoustic-cosine of the clone outputs vs af_heart's own outputs (set later).
    macro_rules! measure {
        ($voice:expr, $self_ref:expr, $label:expr, $af:expr) => {{
            let mut sim = 0.0;
            for (i, t) in HELD_OUT.iter().enumerate() {
                let ids = to_ids(t);
                // GPU synth (the heavy hifigan generator). Drain the queue and give it a long
                // cool-down so the integrated GPU never stays saturated.
                // block_on returns only after the GPU readback completes → queue already drained.
                let out = pollster::block_on(
                    st.synthesize_gpu(&ctx, &pipes, &mut wc, &ids, $voice, None, None),
                );
                cool();
                let s = enc!(&out);
                sim += cosine(acoustic(&s), acoustic(&$af[i]));
            }
            sim /= HELD_OUT.len() as f32;
            let self_sim = cosine(acoustic($self_ref), acoustic($voice));
            println!(
                "  {:<30} clone-vs-af_heart = {:.4}   (self {:.4})",
                $label, sim, self_sim
            );
        }};
    }

    println!(
        "== A0 clone-fidelity calibration ==\nidentity = cosine of the acoustic(timbre) half of the StyleTTS2 style\n"
    );

    // held-out af_heart outputs (the comparison target). Done first = cheap.
    println!(
        "synthesizing {} held-out af_heart outputs (ground truth)...",
        HELD_OUT.len()
    );
    let af_out_styles: Vec<Vec<f32>> = HELD_OUT
        .iter()
        .map(|t| enc!(&ksynth(t, "af_heart")))
        .collect();

    // FLOOR: a different Kokoro speaker (synth'd EARLY so a missing voice panics before the
    // expensive corpus). Compared vs af_heart's outputs.
    println!(
        "synthesizing {} held-out {diff_voice} outputs (floor)...",
        HELD_OUT.len()
    );
    let diff_out_styles: Vec<Vec<f32>> = HELD_OUT
        .iter()
        .map(|t| enc!(&ksynth(t, &diff_voice)))
        .collect();
    let floor = {
        let mut s = 0.0;
        let mut n = 0;
        for d in &diff_out_styles {
            for a in &af_out_styles {
                s += cosine(acoustic(d), acoustic(a));
                n += 1;
            }
        }
        s / n as f32
    };

    // clean varied af_heart reference corpus → per-clip style (the cloning input)
    println!(
        "synthesizing + encoding {} clean af_heart reference clips...",
        REF_CORPUS.len()
    );
    let mut ref_secs: Vec<f32> = Vec::new();
    let mut ref_styles: Vec<Vec<f32>> = Vec::new();
    let mut clip0: Vec<f32> = Vec::new();
    for (i, t) in REF_CORPUS.iter().take(MAX_REF).enumerate() {
        let pcm = ksynth(t, "af_heart");
        ref_secs.push(pcm.len() as f32 / SR);
        if i == 0 {
            clip0 = pcm.clone();
        }
        ref_styles.push(enc!(&pcm));
    }
    let total_ref: f32 = ref_secs.iter().sum();
    let ref_centroid = trimmed_mean(&ref_styles);
    let ceiling: f32 = af_out_styles
        .iter()
        .map(|s| cosine(acoustic(s), acoustic(&ref_centroid)))
        .sum::<f32>()
        / af_out_styles.len() as f32;

    println!(
        "\n  total clean reference available: {total_ref:.1}s ({} clips)",
        ref_styles.len()
    );
    println!("  CEILING (af_heart vs af_heart, diff text) = {ceiling:.4}");
    println!("  FLOOR   ({diff_voice} vs af_heart)        = {floor:.4}");
    println!("  (a faithful clone lands near CEILING, far above FLOOR)\n");

    println!("DATA SWEEP — cloned voice = trimmed-mean of the first N seconds of clean af_heart:");
    for &target in &[12.0f32, 60.0, total_ref] {
        if target > total_ref + 0.1 {
            continue;
        }
        let mut acc = 0.0;
        let mut idx = 0;
        while idx < ref_styles.len() && acc < target {
            acc += ref_secs[idx];
            idx += 1;
        }
        let voice = trimmed_mean(&ref_styles[..idx]);
        measure!(
            &voice,
            &ref_centroid,
            format!("{acc:>5.0}s clean ({idx} clips)"),
            af_out_styles
        );
    }

    println!("\nCONTROLS:");
    // today's path: one short NOISY clip (additive white noise ~-12 dB rel. RMS)
    {
        let mut noisy = clip0.clone();
        let rms = (noisy.iter().map(|x| x * x).sum::<f32>() / noisy.len() as f32).sqrt();
        let mut seed = 0x1234_5678u64;
        for x in noisy.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (seed >> 33) as f32 / (1u32 << 31) as f32 - 1.0;
            *x += u * rms * 0.25;
        }
        let v = enc!(&noisy);
        measure!(
            &v,
            &ref_centroid,
            "1 short NOISY clip".to_string(),
            af_out_styles
        );
    }
    // sanity: cloning a DIFFERENT speaker should land near FLOOR
    {
        let v = enc!(&ksynth(HELD_OUT[0], &diff_voice));
        measure!(
            &v,
            &ref_centroid,
            format!("{diff_voice} clone (≈FLOOR)"),
            af_out_styles
        );
    }

    println!(
        "\nREAD: if the multi-minute clean rows climb toward CEILING ({ceiling:.3}) and sit well"
    );
    println!(
        "above FLOOR ({floor:.3}), the pipeline preserves identity → Phase A (clean data) is the lever."
    );
    println!("The seconds→similarity climb is the real 'how much audio do I need' answer.");
}
