//! Integration tests for the local training path.
//!
//! These tests are gated behind the `local` feature flag and exercise the
//! adapter layer definitions (LoRA, QLoRA, DoRA), quantization utilities,
//! dataset loading, checkpointing, LR scheduling, export config, and the
//! `TrainingBackend` trait surface.

#[cfg(feature = "local")]
mod local_training {
    use rullama_finetune::shared::config::{
        AdapterMethod, AlignmentMethod, LoraConfig, LrScheduler, TrainingHyperparams,
    };
    use rullama_finetune::adapters::{DoraLayer, LoraLayer, QLoraLayer};
    use rullama_finetune::checkpointing::{CheckpointManager, CheckpointMeta};
    use rullama_finetune::dataset_loader::{
        SimpleTokenizer, Tokenizer, TrainingDataset, TrainingExample,
    };
    use rullama_finetune::lr_schedule::LrSchedule;
    use rullama_finetune::quantization::{
        QuantConfig, dequantize_tensor, quantize_tensor,
    };
    use rullama_finetune::{ComputeDevice, LocalTrainingConfig};

    // ── LocalTrainingConfig ──────────────────────────────────────────────

    #[test]
    fn test_local_training_config_new() {
        let cfg = LocalTrainingConfig::new("/model", "/data/train.jsonl", "/output");
        assert_eq!(cfg.model_path.to_str().unwrap(), "/model");
        assert_eq!(cfg.dataset_path.to_str().unwrap(), "/data/train.jsonl");
        assert_eq!(cfg.output_dir.to_str().unwrap(), "/output");
        assert!(cfg.validation_path.is_none());
        assert!(cfg.tokenizer_path.is_none());
        assert_eq!(cfg.device, ComputeDevice::Cpu);
        assert!(cfg.gradient_checkpointing);
        assert!(!cfg.mixed_precision);
    }

    #[test]
    fn test_local_training_config_with_device() {
        let cfg = LocalTrainingConfig::new("/m", "/d", "/o").with_device(ComputeDevice::Gpu {
            index: 0,
            name: "RTX 4090".to_string(),
            vram_mb: 24576,
        });
        assert!(matches!(cfg.device, ComputeDevice::Gpu { index: 0, .. }));
    }

    #[test]
    fn test_local_training_config_with_validation() {
        let cfg = LocalTrainingConfig::new("/m", "/d", "/o").with_validation("/data/val.jsonl");
        assert!(cfg.validation_path.is_some());
    }

    #[test]
    fn test_local_training_config_with_tokenizer() {
        let cfg = LocalTrainingConfig::new("/m", "/d", "/o").with_tokenizer("/tok/tokenizer.json");
        assert!(cfg.tokenizer_path.is_some());
    }

    // ── ComputeDevice ────────────────────────────────────────────────────

    #[test]
    fn test_compute_device_display() {
        assert_eq!(format!("{}", ComputeDevice::Cpu), "CPU");
        assert_eq!(format!("{}", ComputeDevice::Mps), "MPS (Apple Metal)");

        let gpu = ComputeDevice::Gpu {
            index: 0,
            name: "A100".to_string(),
            vram_mb: 40960,
        };
        let display = format!("{}", gpu);
        assert!(display.contains("A100"));
        assert!(display.contains("40960"));
    }

    #[test]
    fn test_compute_device_equality() {
        assert_eq!(ComputeDevice::Cpu, ComputeDevice::Cpu);
        assert_eq!(ComputeDevice::Mps, ComputeDevice::Mps);
        assert_ne!(ComputeDevice::Cpu, ComputeDevice::Mps);
    }

    // ── Adapter layers ───────────────────────────────────────────────────

    #[test]
    fn test_lora_layer_builder_pattern() {
        let layer = LoraLayer::new(512, 512, 8, 16.0).with_dropout(0.1);
        assert_eq!(layer.in_features, 512);
        assert_eq!(layer.out_features, 512);
        assert_eq!(layer.rank, 8);
        assert!((layer.alpha - 16.0).abs() < f32::EPSILON);
        assert!((layer.dropout - 0.1).abs() < f32::EPSILON);
        assert!(layer.active);
    }

    #[test]
    fn test_lora_layer_scaling_various_ranks() {
        for rank in [4, 8, 16, 32, 64] {
            let layer = LoraLayer::new(4096, 4096, rank, rank as f32 * 2.0);
            assert!((layer.scaling() - 2.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_lora_layer_compression_ratio() {
        let layer = LoraLayer::new(4096, 4096, 16, 32.0);
        let ratio = layer.compression_ratio();
        // LoRA should be highly parameter-efficient.
        assert!(ratio < 0.01, "compression ratio {} too high", ratio);
        assert!(ratio > 0.0, "compression ratio must be positive");
    }

    #[test]
    fn test_qlora_vram_savings() {
        let layer_4bit = QLoraLayer::int4(4096, 4096, 16, 32.0);
        let layer_8bit = QLoraLayer::int8(4096, 4096, 16, 32.0);

        // 4-bit should save more VRAM than 8-bit.
        assert!(layer_4bit.vram_savings_ratio() > layer_8bit.vram_savings_ratio());
        // Both should save at least 40%.
        assert!(layer_4bit.vram_savings_ratio() > 0.4);
        assert!(layer_8bit.vram_savings_ratio() > 0.2);
    }

    #[test]
    fn test_qlora_quantized_base_bytes() {
        let layer = QLoraLayer::int4(1024, 1024, 8, 16.0);
        let base_bytes = layer.quantized_base_bytes();
        let full_fp16_bytes = 1024 * 1024 * 2;
        // INT4 should use significantly less than FP16.
        assert!(base_bytes < full_fp16_bytes);
    }

    #[test]
    fn test_dora_layer_trainable_params() {
        let layer = DoraLayer::new(4096, 4096, 16, 32.0);
        let lora_params = 16 * 4096 + 4096 * 16;
        let magnitude_params = 4096;
        assert_eq!(layer.trainable_params(), lora_params + magnitude_params);
    }

    #[test]
    fn test_dora_has_more_params_than_lora() {
        let lora = LoraLayer::new(4096, 4096, 16, 32.0);
        let dora = DoraLayer::new(4096, 4096, 16, 32.0);
        // DoRA has LoRA params + magnitude vector.
        assert!(dora.trainable_params() > lora.trainable_params());
        assert_eq!(
            dora.trainable_params() - lora.trainable_params(),
            4096, // magnitude vector
        );
    }

    // ── Quantization roundtrip ───────────────────────────────────────────

    #[test]
    fn test_quantize_dequantize_roundtrip_int8() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
        let config = QuantConfig::int8();

        let (quantized, scales, zero_points) = quantize_tensor(&data, &config);
        let recovered = dequantize_tensor(&quantized, &scales, &zero_points, config.group_size);

        assert_eq!(recovered.len(), data.len());
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 0.05,
                "INT8 roundtrip error too large: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip_int4() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
        let config = QuantConfig::int4();

        let (quantized, scales, zero_points) = quantize_tensor(&data, &config);
        let recovered = dequantize_tensor(&quantized, &scales, &zero_points, config.group_size);

        assert_eq!(recovered.len(), data.len());
        // INT4 is less precise — allow larger error.
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 0.15,
                "INT4 roundtrip error too large: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_quantize_all_same_values() {
        let data = vec![0.5f32; 64];
        let config = QuantConfig::int4();
        let (quantized, scales, zero_points) = quantize_tensor(&data, &config);
        let recovered = dequantize_tensor(&quantized, &scales, &zero_points, config.group_size);
        for rec in &recovered {
            assert!((rec - 0.5).abs() < 0.01);
        }
    }

    // ── Dataset loading ──────────────────────────────────────────────────

    #[test]
    fn test_dataset_loading_prompt_completion() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("train.jsonl");
        std::fs::write(
            &path,
            r#"{"prompt": "What is Rust?", "completion": "A systems programming language."}
{"prompt": "Who created Rust?", "completion": "Graydon Hoare at Mozilla."}
"#,
        )
        .unwrap();

        let ds = TrainingDataset::load_jsonl(&path).unwrap();
        assert_eq!(ds.len(), 2);
        assert!(!ds.is_empty());
        assert_eq!(ds.steps_per_epoch(1), 2);
        assert_eq!(ds.steps_per_epoch(2), 1);
    }

    #[test]
    fn test_dataset_get_batch() {
        let ds = TrainingDataset {
            examples: (0..10)
                .map(|i| TrainingExample {
                    prompt: format!("prompt{}", i),
                    completion: format!("completion{}", i),
                })
                .collect(),
        };
        let batch = ds.get_batch(3, 4);
        assert_eq!(batch.len(), 4);
        assert_eq!(batch[0].prompt, "prompt3");
        assert_eq!(batch[3].prompt, "prompt6");

        // Batch past end should be truncated.
        let tail = ds.get_batch(8, 5);
        assert_eq!(tail.len(), 2);
    }

    // ── Tokenizer ────────────────────────────────────────────────────────

    #[test]
    fn test_simple_tokenizer_max_seq_len() {
        let tok = SimpleTokenizer::new(5);
        let tokens = tok.encode("Hello, World!");
        assert_eq!(tokens.len(), 5, "should truncate to max_seq_len");
    }

    #[test]
    fn test_simple_tokenizer_encode_example_masking() {
        let tok = SimpleTokenizer::new(1024);
        let example = TrainingExample {
            prompt: "ABC".to_string(),
            completion: "XY".to_string(),
        };
        let (input, target) = tok.encode_example(&example);
        assert_eq!(input.len(), 5); // 3 + 2
        // First 3 tokens should be masked (prompt).
        assert_eq!(target[0], u32::MAX);
        assert_eq!(target[1], u32::MAX);
        assert_eq!(target[2], u32::MAX);
        // Last 2 tokens should match input (completion).
        assert_eq!(target[3], input[3]);
        assert_eq!(target[4], input[4]);
    }

    // ── LR Schedule ──────────────────────────────────────────────────────

    #[test]
    fn test_lr_schedule_cosine_warm_restarts() {
        let sched = LrSchedule::new(1e-3, 0, 100, LrScheduler::CosineWarmRestarts);
        let lr_start = sched.get_lr(1);
        let lr_mid = sched.get_lr(25);
        // Should restart after half the decay period.
        let lr_restart = sched.get_lr(51);

        assert!(lr_start > 0.0);
        assert!(lr_mid < lr_start);
        // After restart, LR should be close to the initial value again.
        assert!(
            (lr_restart - lr_start).abs() < 0.1 * lr_start,
            "restart lr {} should be close to start lr {}",
            lr_restart,
            lr_start
        );
    }

    #[test]
    fn test_lr_schedule_zero_at_step_zero() {
        for scheduler in [
            LrScheduler::Constant,
            LrScheduler::Linear,
            LrScheduler::Cosine,
            LrScheduler::CosineWarmRestarts,
        ] {
            let sched = LrSchedule::new(1e-3, 10, 100, scheduler);
            assert_eq!(
                sched.get_lr(0),
                0.0,
                "LR at step 0 should be 0 for {:?}",
                scheduler
            );
        }
    }

    // ── Checkpointing ────────────────────────────────────────────────────

    #[test]
    fn test_checkpoint_save_load_meta() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = CheckpointManager::new(dir.path());

        let meta = CheckpointMeta {
            epoch: 2,
            step: 500,
            train_loss: 0.123,
            eval_loss: Some(0.145),
            learning_rate: 1e-5,
            timestamp: chrono::Utc::now(),
        };

        mgr.save_meta(500, &meta).unwrap();

        let loaded = CheckpointManager::load_meta(&mgr.checkpoint_path(500)).unwrap();
        assert_eq!(loaded.epoch, 2);
        assert_eq!(loaded.step, 500);
        assert!((loaded.train_loss - 0.123).abs() < f64::EPSILON);
        assert_eq!(loaded.eval_loss, Some(0.145));
    }

    #[test]
    fn test_checkpoint_cleanup_keeps_max() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = CheckpointManager::new(dir.path())
            .with_max_checkpoints(2)
            .with_save_every_steps(100);

        for step in [100, 200, 300, 400] {
            let meta = CheckpointMeta {
                epoch: 1,
                step,
                train_loss: 0.1,
                eval_loss: None,
                learning_rate: 1e-4,
                timestamp: chrono::Utc::now(),
            };
            mgr.save_meta(step, &meta).unwrap();
        }

        let checkpoints = mgr.list_checkpoints();
        assert!(
            checkpoints.len() <= 2,
            "should keep at most 2 checkpoints, found {}",
            checkpoints.len()
        );
    }

    #[test]
    fn test_checkpoint_latest() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = CheckpointManager::new(dir.path())
            .with_max_checkpoints(5)
            .with_save_every_steps(50);

        for step in [50, 100, 150] {
            let meta = CheckpointMeta {
                epoch: 1,
                step,
                train_loss: 0.1,
                eval_loss: None,
                learning_rate: 1e-4,
                timestamp: chrono::Utc::now(),
            };
            mgr.save_meta(step, &meta).unwrap();
        }

        let latest = mgr.latest_checkpoint().unwrap();
        assert!(
            latest.to_str().unwrap().contains("checkpoint-150"),
            "latest checkpoint should be at step 150, got {:?}",
            latest
        );
    }

    #[test]
    fn test_checkpoint_save_load_weights_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = CheckpointManager::new(dir.path());

        let mut weights = std::collections::HashMap::new();
        weights.insert(
            "layer.lora_a".to_string(),
            (vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], vec![2, 4]),
        );
        weights.insert(
            "layer.lora_b".to_string(),
            (vec![0.9f32, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], vec![4, 2]),
        );

        mgr.save_weights(200, &weights).unwrap();

        let loaded = CheckpointManager::load_weights(&mgr.checkpoint_path(200)).unwrap();
        assert_eq!(loaded.len(), 2);

        let (data, shape) = &loaded["layer.lora_a"];
        assert_eq!(shape, &[2, 4]);
        assert!((data[0] - 0.1).abs() < 1e-6);
        assert!((data[7] - 0.8).abs() < 1e-6);
    }

    // ── AdapterMethod config ─────────────────────────────────────────────

    #[test]
    fn test_adapter_method_serialization_roundtrip() {
        let methods = vec![
            AdapterMethod::LoRA,
            AdapterMethod::QLoRA { bits: 4 },
            AdapterMethod::QLoRA { bits: 8 },
            AdapterMethod::DoRA,
            AdapterMethod::QDoRA { bits: 4 },
        ];

        for method in methods {
            let json = serde_json::to_string(&method).unwrap();
            let parsed: AdapterMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, method, "roundtrip failed for {:?}", method);
        }
    }

    #[test]
    fn test_alignment_method_constructors() {
        let dpo = AlignmentMethod::dpo();
        assert!(matches!(dpo, AlignmentMethod::DPO { .. }));

        let orpo = AlignmentMethod::orpo();
        assert!(matches!(orpo, AlignmentMethod::ORPO { .. }));

        let none = AlignmentMethod::default();
        assert!(matches!(none, AlignmentMethod::None));
    }

    // ── Export config (struct only, no GPU required) ─────────────────────

    #[test]
    fn test_export_format_display() {
        use rullama_finetune::export::ExportFormat;
        assert_eq!(format!("{}", ExportFormat::Gguf), "gguf");
        assert_eq!(format!("{}", ExportFormat::SafeTensors), "safetensors");
        assert_eq!(format!("{}", ExportFormat::AdapterOnly), "adapter_only");
    }

    #[test]
    fn test_export_config_gguf() {
        use rullama_finetune::export::ExportConfig;
        let cfg = ExportConfig::gguf("/output/model.gguf");
        assert_eq!(
            cfg.gguf_quantization.as_deref(),
            Some("Q4_K_M"),
            "default GGUF quantization should be Q4_K_M"
        );
        assert!(cfg.include_metadata);
    }

    // ── TrainingBackend trait (BurnBackend struct check) ──────────────────

    #[test]
    fn test_burn_backend_instantiation() {
        use rullama_finetune::BurnBackend;
        let _backend = BurnBackend::new();
        // Just verifying the struct can be created without panicking.
    }

    // ── LoraConfig default targets ───────────────────────────────────────

    #[test]
    fn test_default_lora_targets_all_attention_projections() {
        let config = LoraConfig::default();
        assert!(config.target_modules.contains(&"q_proj".to_string()));
        assert!(config.target_modules.contains(&"k_proj".to_string()));
        assert!(config.target_modules.contains(&"v_proj".to_string()));
        assert!(config.target_modules.contains(&"o_proj".to_string()));
    }

    // ── Hyperparams with custom values ───────────────────────────────────

    #[test]
    fn test_hyperparams_serialization() {
        let hp = TrainingHyperparams {
            epochs: 5,
            batch_size: 8,
            learning_rate: 1e-4,
            warmup_steps: 200,
            weight_decay: 0.05,
            lr_scheduler: LrScheduler::CosineWarmRestarts,
            seed: 123,
            max_seq_len: 4096,
            gradient_accumulation_steps: 8,
            max_grad_norm: 0.5,
        };
        let json = serde_json::to_string(&hp).unwrap();
        let parsed: TrainingHyperparams = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.epochs, 5);
        assert_eq!(parsed.lr_scheduler, LrScheduler::CosineWarmRestarts);
        assert_eq!(parsed.max_seq_len, 4096);
    }
}
