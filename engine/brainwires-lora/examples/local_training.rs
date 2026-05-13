//! Example: Local LoRA/QLoRA training pipeline
//!
//! Demonstrates how to configure a local training run with adapter
//! parameters, compute device selection, and training progress tracking.
//!
//! Run: cargo run -p rullama-finetune --example local_training

use std::path::PathBuf;

use rullama_finetune::shared::config::{
    AdapterMethod, AlignmentMethod, LoraConfig, LrScheduler, TrainingHyperparams,
};
use rullama_finetune::{BurnBackend, ComputeDevice, LocalTrainingConfig, TrainingBackend};

fn main() {
    // Step 1: Choose a compute device.
    // In a real application you would query the backend for available devices.
    let device = ComputeDevice::Gpu {
        index: 0,
        name: "NVIDIA RTX 4090".to_string(),
        vram_mb: 24_576,
    };
    println!("Selected device: {}", device);

    // Step 2: Configure LoRA adapter parameters.
    // QLoRA uses 4-bit quantized base weights to reduce VRAM usage.
    let lora = LoraConfig {
        rank: 32,
        alpha: 64.0,
        dropout: 0.05,
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
        ],
        method: AdapterMethod::QLoRA { bits: 4 },
    };

    println!(
        "Adapter: {:?} (rank={}, alpha={}, quantized={})",
        lora.method,
        lora.rank,
        lora.alpha,
        lora.method.is_quantized(),
    );

    // Step 3: Set training hyperparameters.
    let hyperparams = TrainingHyperparams {
        epochs: 3,
        batch_size: 2,
        learning_rate: 2e-4,
        warmup_steps: 100,
        weight_decay: 0.01,
        lr_scheduler: LrScheduler::Cosine,
        seed: 42,
        max_seq_len: 2048,
        gradient_accumulation_steps: 8,
        max_grad_norm: 1.0,
    };

    // Step 4: Build the local training configuration.
    // Paths point to model weights, training data, and output directory.
    let mut config = LocalTrainingConfig::new(
        PathBuf::from("/models/llama-3-8b"),
        PathBuf::from("/data/train.jsonl"),
        PathBuf::from("/output/llama-3-8b-lora"),
    )
    .with_device(device)
    .with_validation("/data/val.jsonl")
    .with_tokenizer("/models/llama-3-8b/tokenizer.json");

    config.hyperparams = hyperparams;
    config.lora = lora;
    config.alignment = AlignmentMethod::dpo();
    config.gradient_checkpointing = true;
    config.mixed_precision = true;

    println!("\nTraining configuration:");
    println!("  Model path:     {:?}", config.model_path);
    println!("  Dataset path:   {:?}", config.dataset_path);
    println!("  Output dir:     {:?}", config.output_dir);
    println!("  Epochs:         {}", config.hyperparams.epochs);
    println!("  Batch size:     {}", config.hyperparams.batch_size);
    println!("  Learning rate:  {}", config.hyperparams.learning_rate);
    println!(
        "  Grad accum:     {}",
        config.hyperparams.gradient_accumulation_steps
    );
    println!("  Mixed precision: {}", config.mixed_precision);
    println!("  Grad checkpointing: {}", config.gradient_checkpointing);

    // Step 5: Pick a training backend. The `TrainingBackend` trait lets you
    // swap implementations; rullama-finetune ships `BurnBackend`.
    let backend = BurnBackend::default();
    println!("\nBackend: {}", backend.name());
    println!("Backend devices:");
    for d in backend.available_devices() {
        println!("  - {}", d);
    }

    // The actual training call would look like:
    //
    //   let artifact = backend.train(config, Box::new(|progress| {
    //       println!(
    //           "Step {}/{} | epoch {}/{} | loss: {:.4}",
    //           progress.step,
    //           progress.total_steps,
    //           progress.epoch,
    //           progress.total_epochs,
    //           progress.train_loss.unwrap_or(0.0),
    //       );
    //   })).unwrap();
    //
    //   println!("Saved to: {:?} (format: {})", artifact.model_path, artifact.format);

    // Step 6: Demonstrate ComputeDevice variants.
    let devices = vec![
        ComputeDevice::Cpu,
        ComputeDevice::Gpu {
            index: 0,
            name: "NVIDIA RTX 4090".to_string(),
            vram_mb: 24_576,
        },
        ComputeDevice::Mps,
    ];

    println!("\nAvailable device types:");
    for d in &devices {
        println!("  - {}", d);
    }

    let _ = config; // suppress unused-variable warning (training call is commented out)

    println!("\nLocal training pipeline configured successfully.");
}
