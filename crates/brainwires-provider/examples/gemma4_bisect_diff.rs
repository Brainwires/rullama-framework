//! Bisect diff tool — pairs up per-layer checkpoint dumps from two
//! Gemma 4 inference runs (e.g. CPU vs WGPU) and reports per-checkpoint
//! max-abs-diff + L2 distance, sorted to highlight the FIRST layer
//! where outputs diverge.
//!
//! The producer that wrote these dumps (the candle-based `gemma4_diag`
//! example) was moved out of this workspace to `rullama` along with
//! the rest of the low-level Candle inference path. This bisect tool
//! is preserved here because it's a standalone f32-blob diff utility
//! with no candle/wgpu dependency — point it at two dump directories
//! produced by any tool that writes the file format below.
//!
//! ## How to use
//!
//! ```sh
//! cargo run --example gemma4_bisect_diff -- /tmp/g4-cpu /tmp/g4-wgpu
//! ```
//!
//! The tool scans both dirs for matching files
//! (filename = `step{NNNN}_layer{MMM}_{label}.bin`) and reports:
//!    - max-abs-diff per checkpoint
//!    - L2 distance per checkpoint
//!    - which layer + checkpoint diverged FIRST (most useful info)
//!    - top-5 worst-divergence checkpoints
//!
//! ## File format
//!
//! Each `.bin` file:
//!   - 4-byte magic `BST1`
//!   - 1-byte rank
//!   - rank × 8-byte LE u64 dim sizes
//!   - 1-byte dtype tag (always 0 = F32 for now)
//!   - rest = LE f32 values (rows × cols × ...)

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CheckpointFile {
    path: PathBuf,
    step: u32,
    layer: u32,
    label: String,
    shape: Vec<u64>,
    data: Vec<f32>,
}

fn parse_filename(path: &Path) -> Option<(u32, u32, String)> {
    let name = path.file_name()?.to_str()?;
    if !name.ends_with(".bin") {
        return None;
    }
    let stem = name.strip_suffix(".bin")?;
    // step{NNNN}_layer{MMM}_{label}
    let mut parts = stem.splitn(3, '_');
    let step_part = parts.next()?;
    let layer_part = parts.next()?;
    let label = parts.next()?.to_string();
    let step = step_part.strip_prefix("step")?.parse().ok()?;
    let layer = layer_part.strip_prefix("layer")?.parse().ok()?;
    Some((step, layer, label))
}

fn read_dump(path: &Path) -> std::io::Result<CheckpointFile> {
    let mut f = File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"BST1" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad magic",
        ));
    }
    let mut rank_buf = [0u8; 1];
    f.read_exact(&mut rank_buf)?;
    let rank = rank_buf[0] as usize;
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        let mut dim_buf = [0u8; 8];
        f.read_exact(&mut dim_buf)?;
        shape.push(u64::from_le_bytes(dim_buf));
    }
    let mut dtype_buf = [0u8; 1];
    f.read_exact(&mut dtype_buf)?;
    if dtype_buf[0] != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported dtype",
        ));
    }
    let n_elem: u64 = shape.iter().product();
    let mut raw = vec![0u8; (n_elem * 4) as usize];
    f.read_exact(&mut raw)?;
    let mut data = Vec::with_capacity(n_elem as usize);
    for chunk in raw.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let (step, layer, label) = parse_filename(path)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad filename"))?;
    Ok(CheckpointFile {
        path: path.to_path_buf(),
        step,
        layer,
        label,
        shape,
        data,
    })
}

fn list_dumps(dir: &Path) -> std::io::Result<Vec<CheckpointFile>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) == Some("bin") {
            match read_dump(&p) {
                Ok(cf) => out.push(cf),
                Err(e) => eprintln!("warn: skipping {} ({})", p.display(), e),
            }
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Diff {
    step: u32,
    layer: u32,
    label: String,
    n_elem: usize,
    a_l2: f64,
    b_l2: f64,
    max_abs_diff: f64,
    l2_diff: f64,
    relative: f64,
    nan_a: usize,
    nan_b: usize,
}

fn diff_pair(a: &CheckpointFile, b: &CheckpointFile) -> Diff {
    let n = a.data.len().min(b.data.len());
    let mut max_abs = 0.0_f64;
    let mut sum_sq_a = 0.0_f64;
    let mut sum_sq_b = 0.0_f64;
    let mut sum_sq_d = 0.0_f64;
    let mut nan_a = 0usize;
    let mut nan_b = 0usize;
    for i in 0..n {
        let av = a.data[i];
        let bv = b.data[i];
        if !av.is_finite() {
            nan_a += 1;
        }
        if !bv.is_finite() {
            nan_b += 1;
        }
        if av.is_finite() && bv.is_finite() {
            let d = (av - bv) as f64;
            sum_sq_a += (av as f64) * (av as f64);
            sum_sq_b += (bv as f64) * (bv as f64);
            sum_sq_d += d * d;
            let ad = d.abs();
            if ad > max_abs {
                max_abs = ad;
            }
        }
    }
    let a_l2 = sum_sq_a.sqrt();
    let b_l2 = sum_sq_b.sqrt();
    let l2_diff = sum_sq_d.sqrt();
    let denom = a_l2.max(b_l2).max(1e-12);
    let relative = l2_diff / denom;
    Diff {
        step: a.step,
        layer: a.layer,
        label: a.label.clone(),
        n_elem: n,
        a_l2,
        b_l2,
        max_abs_diff: max_abs,
        l2_diff,
        relative,
        nan_a,
        nan_b,
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: {} <dir-a> <dir-b> [--threshold <rel>]\n\n\
             Compares two CANDLE_BISECT_DUMP_DIR outputs. Reports the first\n\
             checkpoint where relative L2 difference exceeds the threshold\n\
             (default 0.001 = 0.1%) and the top-5 worst checkpoints.",
            args.first()
                .map(|s| s.as_str())
                .unwrap_or("gemma4_bisect_diff"),
        );
        return ExitCode::from(2);
    }
    let dir_a = PathBuf::from(&args[1]);
    let dir_b = PathBuf::from(&args[2]);
    let mut threshold = 0.001_f64;
    let mut layer_filter: Option<u32> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--threshold" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse::<f64>().ok()) {
                    threshold = v;
                    i += 2;
                    continue;
                }
            }
            "--layer" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse::<u32>().ok()) {
                    layer_filter = Some(v);
                    i += 2;
                    continue;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let a = match list_dumps(&dir_a) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", dir_a.display(), e);
            return ExitCode::from(1);
        }
    };
    let b = match list_dumps(&dir_b) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", dir_b.display(), e);
            return ExitCode::from(1);
        }
    };

    // Index by (step, layer, label).
    let mut by_key: BTreeMap<(u32, u32, String), (Option<CheckpointFile>, Option<CheckpointFile>)> =
        BTreeMap::new();
    for cf in a {
        let key = (cf.step, cf.layer, cf.label.clone());
        by_key.entry(key).or_insert((None, None)).0 = Some(cf);
    }
    for cf in b {
        let key = (cf.step, cf.layer, cf.label.clone());
        by_key.entry(key).or_insert((None, None)).1 = Some(cf);
    }

    let mut diffs: Vec<Diff> = Vec::new();
    let mut a_only = 0usize;
    let mut b_only = 0usize;
    let mut shape_mismatch = 0usize;
    for ((_step, _layer, _label), (oa, ob)) in by_key.iter() {
        match (oa, ob) {
            (Some(a), Some(b)) => {
                if a.shape != b.shape {
                    shape_mismatch += 1;
                    eprintln!(
                        "warn: shape mismatch step={} layer={} label={}: {:?} vs {:?}",
                        a.step, a.layer, a.label, a.shape, b.shape,
                    );
                    continue;
                }
                diffs.push(diff_pair(a, b));
            }
            (Some(_), None) => a_only += 1,
            (None, Some(_)) => b_only += 1,
            _ => {}
        }
    }

    println!(
        "==> Bisect diff: {} ↔ {}\n    matched checkpoints: {}, only-A: {}, only-B: {}, shape-mismatch: {}",
        dir_a.display(),
        dir_b.display(),
        diffs.len(),
        a_only,
        b_only,
        shape_mismatch,
    );

    // First divergence (sorted by step, layer, label which is BTreeMap order).
    let first_diverge = diffs.iter().find(|d| d.relative > threshold);
    println!(
        "\n==> First divergence above threshold (rel L2 > {})",
        threshold
    );
    if let Some(d) = first_diverge {
        println!(
            "    step={:4} layer={:3} {:30} | max|Δ|={:.4e} L2_diff={:.4e} rel={:.4e}",
            d.step, d.layer, d.label, d.max_abs_diff, d.l2_diff, d.relative,
        );
    } else {
        println!(
            "    NO checkpoint diverged above threshold ({}). Devices match within tolerance.",
            threshold
        );
    }

    // Top 10 by relative diff.
    let mut sorted = diffs.clone();
    sorted.sort_by(|a, b| {
        b.relative
            .partial_cmp(&a.relative)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!("\n==> Top-10 worst divergences (by relative L2)");
    println!(
        "    {:>4} {:>3} {:30} {:>11} {:>11} {:>11} {:>10} {:>10}",
        "step", "lyr", "label", "max|Δ|", "L2(diff)", "rel", "L2(A)", "L2(B)",
    );
    for d in sorted.iter().take(10) {
        println!(
            "    {:>4} {:>3} {:30} {:>11.3e} {:>11.3e} {:>11.3e} {:>10.3e} {:>10.3e}",
            d.step, d.layer, d.label, d.max_abs_diff, d.l2_diff, d.relative, d.a_l2, d.b_l2,
        );
    }

    // Layer-wise summary — average relative diff per (step, layer) across all
    // checkpoints in that layer. Highlights "everything in layer X drifts".
    let mut by_layer: BTreeMap<(u32, u32), Vec<f64>> = BTreeMap::new();
    for d in &diffs {
        by_layer
            .entry((d.step, d.layer))
            .or_default()
            .push(d.relative);
    }
    println!("\n==> Per-layer summary (max + mean rel L2 across checkpoints in that layer)");
    println!(
        "    {:>4} {:>3} {:>11} {:>11} {:>5}",
        "step", "lyr", "max-rel", "mean-rel", "n"
    );
    let mut lay_rows: Vec<((u32, u32), f64, f64, usize)> = by_layer
        .iter()
        .map(|((s, l), v)| {
            let max = v.iter().cloned().fold(0.0_f64, f64::max);
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            ((*s, *l), max, mean, v.len())
        })
        .collect();
    lay_rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for ((s, l), max, mean, n) in lay_rows.iter().take(15) {
        println!(
            "    {:>4} {:>3} {:>11.3e} {:>11.3e} {:>5}",
            s, l, max, mean, n
        );
    }

    // Per-layer chain view: show every checkpoint for one specific
    // layer in source order so the propagation of drift is visible.
    if let Some(filter) = layer_filter {
        println!(
            "\n==> Full checkpoint chain for layer {} (sorted by step, label)",
            filter,
        );
        println!(
            "    {:>4} {:30} {:>11} {:>11} {:>11} {:>10}",
            "step", "label", "max|Δ|", "L2(diff)", "rel", "L2(A)",
        );
        let mut chain: Vec<&Diff> = diffs.iter().filter(|d| d.layer == filter).collect();
        chain.sort_by(|a, b| a.step.cmp(&b.step).then_with(|| a.label.cmp(&b.label)));
        for d in chain {
            println!(
                "    {:>4} {:30} {:>11.3e} {:>11.3e} {:>11.3e} {:>10.3e}",
                d.step, d.label, d.max_abs_diff, d.l2_diff, d.relative, d.a_l2,
            );
        }
    }

    // NaN / Inf report.
    let nan_total_a: usize = diffs.iter().map(|d| d.nan_a).sum();
    let nan_total_b: usize = diffs.iter().map(|d| d.nan_b).sum();
    if nan_total_a > 0 || nan_total_b > 0 {
        println!(
            "\n==> NaN/Inf detected: A={}, B={}",
            nan_total_a, nan_total_b
        );
    }

    if first_diverge.is_some() {
        ExitCode::from(1)
    } else {
        ExitCode::from(0)
    }
}
