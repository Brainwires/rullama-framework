//! Validate the real Z-Image VAE decoder structure against the parsed config
//! (single-file safetensors). Confirms ShardedSafetensors::open_single + the
//! decoder's conv/resnet/attn/groupnorm shapes before wiring the VAE forward
//! (IM3). The decoder is a standard diffusers AutoencoderKL:
//!   conv_in → mid_block(resnet, attn, resnet) → up_blocks×4 → norm_out → conv_out
//!
//! Usage:
//!   cargo run -p rullama-engine --example imagegen_vae_inventory -- \
//!       weights/Z-Image-Turbo/vae

use rullama_engine::imagegen::{ShardedSafetensors, VaeConfig};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "weights/Z-Image-Turbo/vae".to_string());

    let cfg = VaeConfig::parse(&std::fs::read(format!("{dir}/config.json")).expect("config"))
        .expect("parse VAE config");
    let boc = &cfg.block_out_channels;
    let (lo, hi) = (boc[0] as usize, *boc.last().unwrap() as usize);
    println!(
        "VAE: latent_ch={} block_out={:?} groups={} downscale={}x mid_attn={}",
        cfg.latent_channels,
        boc,
        cfg.norm_num_groups,
        cfg.downscale(),
        cfg.mid_block_add_attention
    );

    let st = ShardedSafetensors::open_single(format!("{dir}/diffusion_pytorch_model.safetensors"))
        .expect("open VAE safetensors");
    let dec = st.names().filter(|n| n.starts_with("decoder")).count();
    println!("loaded {} tensors ({} decoder)", st.names().count(), dec);

    let lat = cfg.latent_channels as usize;
    let out = cfg.out_channels as usize;

    // Entry/exit convs: conv_in latent→hi (3×3), conv_out lo→rgb (3×3).
    check(&st, "decoder.conv_in.weight", &[hi, lat, 3, 3]);
    check(&st, "decoder.conv_out.weight", &[out, lo, 3, 3]);
    check(&st, "decoder.conv_norm_out.weight", &[lo]);

    // Mid block: 2 resnets + 1 self-attention, all at the top channel width.
    for r in 0..2 {
        let p = format!("decoder.mid_block.resnets.{r}");
        check(&st, &format!("{p}.norm1.weight"), &[hi]);
        check(&st, &format!("{p}.conv1.weight"), &[hi, hi, 3, 3]);
        check(&st, &format!("{p}.conv2.weight"), &[hi, hi, 3, 3]);
    }
    if cfg.mid_block_add_attention {
        let p = "decoder.mid_block.attentions.0";
        check(&st, &format!("{p}.group_norm.weight"), &[hi]);
        for proj in ["to_q", "to_k", "to_v"] {
            check(&st, &format!("{p}.{proj}.weight"), &[hi, hi]);
        }
        check(&st, &format!("{p}.to_out.0.weight"), &[hi, hi]);
    }

    // Up blocks: reversed channels [hi … lo], each with layers_per_block+1
    // resnets; all but the last carry an upsampler conv.
    let rev: Vec<usize> = boc.iter().rev().map(|&c| c as usize).collect();
    let nblocks = rev.len();
    let mut upsamplers = 0;
    for (bi, &ch) in rev.iter().enumerate() {
        let p = format!("decoder.up_blocks.{bi}");
        // first resnet of block 0 maps hi→hi; generally prev_ch→ch.
        let r0 = format!("{p}.resnets.0.conv2.weight");
        check(&st, &r0, &[ch, ch, 3, 3]);
        if st.has(&format!("{p}.upsamplers.0.conv.weight")) {
            upsamplers += 1;
        }
    }
    println!("up_blocks: {nblocks} (channels {rev:?}), {upsamplers} upsamplers");

    // Spot-check a real range-read + dequant.
    let w = st
        .tensor_f32("decoder.conv_out.weight")
        .expect("read conv_out");
    println!(
        "conv_out: {} elems dtype {:?}, mean {:.5}",
        w.len(),
        st.dtype("decoder.conv_out.weight").unwrap(),
        w.iter().sum::<f32>() / w.len() as f32
    );

    println!("\nOK — VAE decoder structure matches config.");
}

fn check(st: &ShardedSafetensors, name: &str, expect: &[usize]) {
    match st.shape(name) {
        Ok(s) if s == expect => {}
        Ok(s) => panic!("{name}: shape {s:?} != expected {expect:?}"),
        Err(e) => panic!("{name}: {e}"),
    }
}
