use std::time::Instant;

use burn_core::module::AutodiffModule;
use burn_optim::{AdamConfig, GradientsParams, Optimizer};
use burn_wgpu::WgpuDevice;
use tracing::{info, warn};

use super::batch::make_batch;
use super::weights::{finalize_training, try_load_quantized_weights, try_load_safetensors_weights};
use crate::shared::error::TrainingError;
use crate::burn_modules::{DoraLinearConfig, LoraLinearConfig, QLoraLinearConfig};
use crate::checkpointing::{CheckpointManager, CheckpointMeta};
use crate::dataset_loader::{Tokenizer, TrainingDataset};
use crate::lr_schedule::LrSchedule;
use crate::weight_loader::SafeTensorsLoader;
use crate::{LocalTrainingConfig, TrainedModelArtifact};
use crate::shared::types::TrainingProgress;

/// Run LoRA fine-tuning with real dataset.
pub(super) fn train_lora(
    config: &LocalTrainingConfig,
    dataset: &TrainingDataset,
    tokenizer: &dyn Tokenizer,
    validation_dataset: Option<&TrainingDataset>,
    callback: &dyn Fn(TrainingProgress),
) -> Result<TrainedModelArtifact, TrainingError> {
    let device = WgpuDevice::default();
    let start = Instant::now();
    let rank = config.lora.rank as usize;
    let dim = SafeTensorsLoader::open(&config.model_path)
        .ok()
        .and_then(|loader| loader.load_config())
        .map(|c| c.hidden_size)
        .unwrap_or(rank * 64);

    info!("Initializing LoRA training on WGPU device");

    let lora_config = LoraLinearConfig::new(dim, dim)
        .with_rank(rank)
        .with_alpha(config.lora.alpha);

    // Try loading base weights from SafeTensors
    let model = if let Some(base_weight) = try_load_safetensors_weights(config, dim, &device) {
        lora_config.init_with_base_weights::<super::types::TrainBackend>(base_weight, &device)
    } else {
        lora_config.init::<super::types::TrainBackend>(&device)
    };

    let batch_size = config.hyperparams.batch_size as usize;
    let steps_per_epoch = dataset.steps_per_epoch(batch_size);
    let total_steps = config.hyperparams.epochs as u64 * steps_per_epoch;

    let lr_schedule = LrSchedule::new(
        config.hyperparams.learning_rate,
        config.hyperparams.warmup_steps,
        total_steps,
        config.hyperparams.lr_scheduler,
    );

    let optim_config = AdamConfig::new().with_weight_decay(Some(
        burn_optim::decay::WeightDecayConfig::new(config.hyperparams.weight_decay as f32),
    ));
    let mut optim = optim_config.init();

    let checkpoint_mgr = CheckpointManager::new(&config.output_dir)
        .with_save_every_steps(500)
        .with_max_checkpoints(3);

    let mut global_step = 0u64;
    let mut model = model;
    let mut running_loss = 0.0f32;

    info!(
        "Training: {} epochs, {} steps/epoch, {} total, lr={}, batch={}",
        config.hyperparams.epochs,
        steps_per_epoch,
        total_steps,
        config.hyperparams.learning_rate,
        batch_size,
    );

    for epoch in 0..config.hyperparams.epochs {
        let epoch_start = Instant::now();

        for step in 0..steps_per_epoch {
            global_step += 1;
            let lr = lr_schedule.get_lr(global_step);

            let batch_start = (step as usize * batch_size) % dataset.len();
            let (input, target) =
                make_batch(dataset, tokenizer, batch_start, batch_size, dim, &device);

            let output = model.forward(input);
            let diff = output - target;
            let loss = diff.clone().powf_scalar(2.0).mean();

            let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap_or_default();
            let loss_scalar = loss_val.first().copied().unwrap_or(0.0);
            running_loss = running_loss * 0.99 + loss_scalar * 0.01;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            if checkpoint_mgr.should_save(global_step) {
                let meta = CheckpointMeta {
                    epoch: epoch + 1,
                    step: global_step,
                    train_loss: running_loss as f64,
                    eval_loss: None,
                    learning_rate: lr,
                    timestamp: chrono::Utc::now(),
                };
                if let Err(e) = checkpoint_mgr.save_meta(global_step, &meta) {
                    warn!("Failed to save checkpoint: {}", e);
                }
            }

            if global_step.is_multiple_of(10) || global_step == total_steps {
                callback(TrainingProgress {
                    epoch: epoch + 1,
                    total_epochs: config.hyperparams.epochs,
                    step: global_step,
                    total_steps,
                    train_loss: Some(running_loss as f64),
                    eval_loss: None,
                    learning_rate: Some(lr),
                    elapsed_secs: start.elapsed().as_secs(),
                });
            }
        }

        // End-of-epoch validation
        let eval_loss = validation_dataset.map(|vd| {
            let vd_steps = vd.steps_per_epoch(batch_size);
            let mut total_loss = 0.0f32;
            for vs in 0..vd_steps {
                let vb_start = (vs as usize * batch_size) % vd.len();
                let (vi, vt) = make_batch(vd, tokenizer, vb_start, batch_size, dim, &device);
                let vo = model.forward(vi);
                let vdiff = vo - vt;
                let vloss = vdiff.clone().powf_scalar(2.0).mean();
                let vl = vloss.into_data().to_vec::<f32>().unwrap_or_default();
                total_loss += vl.first().copied().unwrap_or(0.0);
            }
            let avg = total_loss / vd_steps.max(1) as f32;
            info!(
                "Epoch {}/{} eval_loss: {:.6}",
                epoch + 1,
                config.hyperparams.epochs,
                avg
            );
            avg as f64
        });

        let epoch_duration = epoch_start.elapsed();
        info!(
            "Epoch {}/{} complete in {:.1}s, train_loss: {:.6}{}",
            epoch + 1,
            config.hyperparams.epochs,
            epoch_duration.as_secs_f64(),
            running_loss,
            eval_loss
                .map(|l| format!(", eval_loss: {:.6}", l))
                .unwrap_or_default(),
        );
    }

    let inner = model.valid();
    let a_data = inner.lora_a_weight().into_data();
    let b_data = inner.lora_b_weight().into_data();

    finalize_training(
        config,
        running_loss,
        total_steps,
        &start,
        &a_data.bytes,
        &b_data.bytes,
        None,
    )
}

/// Run DoRA fine-tuning with real dataset.
pub(super) fn train_dora(
    config: &LocalTrainingConfig,
    dataset: &TrainingDataset,
    tokenizer: &dyn Tokenizer,
    validation_dataset: Option<&TrainingDataset>,
    callback: &dyn Fn(TrainingProgress),
) -> Result<TrainedModelArtifact, TrainingError> {
    let device = WgpuDevice::default();
    let start = Instant::now();
    let rank = config.lora.rank as usize;
    let dim = SafeTensorsLoader::open(&config.model_path)
        .ok()
        .and_then(|loader| loader.load_config())
        .map(|c| c.hidden_size)
        .unwrap_or(rank * 64);

    info!("Initializing DoRA training on WGPU device");

    let dora_config = DoraLinearConfig::new(dim, dim)
        .with_rank(rank)
        .with_alpha(config.lora.alpha);

    let model = if let Some(base_weight) = try_load_safetensors_weights(config, dim, &device) {
        dora_config.init_with_base_weights::<super::types::TrainBackend>(base_weight, &device)
    } else {
        dora_config.init::<super::types::TrainBackend>(&device)
    };

    let batch_size = config.hyperparams.batch_size as usize;
    let steps_per_epoch = dataset.steps_per_epoch(batch_size);
    let total_steps = config.hyperparams.epochs as u64 * steps_per_epoch;

    let lr_schedule = LrSchedule::new(
        config.hyperparams.learning_rate,
        config.hyperparams.warmup_steps,
        total_steps,
        config.hyperparams.lr_scheduler,
    );

    let optim_config = AdamConfig::new().with_weight_decay(Some(
        burn_optim::decay::WeightDecayConfig::new(config.hyperparams.weight_decay as f32),
    ));
    let mut optim = optim_config.init();

    let checkpoint_mgr = CheckpointManager::new(&config.output_dir)
        .with_save_every_steps(500)
        .with_max_checkpoints(3);

    let mut global_step = 0u64;
    let mut model = model;
    let mut running_loss = 0.0f32;

    info!(
        "Training: {} epochs, {} steps/epoch, {} total, lr={}, batch={}",
        config.hyperparams.epochs,
        steps_per_epoch,
        total_steps,
        config.hyperparams.learning_rate,
        batch_size,
    );

    for epoch in 0..config.hyperparams.epochs {
        let epoch_start = Instant::now();

        for step in 0..steps_per_epoch {
            global_step += 1;
            let lr = lr_schedule.get_lr(global_step);

            let batch_start = (step as usize * batch_size) % dataset.len();
            let (input, target) =
                make_batch(dataset, tokenizer, batch_start, batch_size, dim, &device);

            let output = model.forward(input);
            let diff = output - target;
            let loss = diff.clone().powf_scalar(2.0).mean();

            let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap_or_default();
            let loss_scalar = loss_val.first().copied().unwrap_or(0.0);
            running_loss = running_loss * 0.99 + loss_scalar * 0.01;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            if checkpoint_mgr.should_save(global_step) {
                let meta = CheckpointMeta {
                    epoch: epoch + 1,
                    step: global_step,
                    train_loss: running_loss as f64,
                    eval_loss: None,
                    learning_rate: lr,
                    timestamp: chrono::Utc::now(),
                };
                if let Err(e) = checkpoint_mgr.save_meta(global_step, &meta) {
                    warn!("Failed to save checkpoint: {}", e);
                }
            }

            if global_step.is_multiple_of(10) || global_step == total_steps {
                callback(TrainingProgress {
                    epoch: epoch + 1,
                    total_epochs: config.hyperparams.epochs,
                    step: global_step,
                    total_steps,
                    train_loss: Some(running_loss as f64),
                    eval_loss: None,
                    learning_rate: Some(lr),
                    elapsed_secs: start.elapsed().as_secs(),
                });
            }
        }

        // End-of-epoch validation
        let eval_loss = validation_dataset.map(|vd| {
            let vd_steps = vd.steps_per_epoch(batch_size);
            let mut total_loss = 0.0f32;
            for vs in 0..vd_steps {
                let vb_start = (vs as usize * batch_size) % vd.len();
                let (vi, vt) = make_batch(vd, tokenizer, vb_start, batch_size, dim, &device);
                let vo = model.forward(vi);
                let vdiff = vo - vt;
                let vloss = vdiff.clone().powf_scalar(2.0).mean();
                let vl = vloss.into_data().to_vec::<f32>().unwrap_or_default();
                total_loss += vl.first().copied().unwrap_or(0.0);
            }
            let avg = total_loss / vd_steps.max(1) as f32;
            info!(
                "Epoch {}/{} eval_loss: {:.6}",
                epoch + 1,
                config.hyperparams.epochs,
                avg
            );
            avg as f64
        });

        let epoch_duration = epoch_start.elapsed();
        info!(
            "Epoch {}/{} complete in {:.1}s, train_loss: {:.6}{}",
            epoch + 1,
            config.hyperparams.epochs,
            epoch_duration.as_secs_f64(),
            running_loss,
            eval_loss
                .map(|l| format!(", eval_loss: {:.6}", l))
                .unwrap_or_default(),
        );
    }

    let inner = model.valid();
    let a_data = inner.lora_a_weight().into_data();
    let b_data = inner.lora_b_weight().into_data();
    let m_data = inner.magnitude_data().into_data();

    finalize_training(
        config,
        running_loss,
        total_steps,
        &start,
        &a_data.bytes,
        &b_data.bytes,
        Some(&m_data.bytes),
    )
}

/// Run QLoRA fine-tuning with quantized base weights.
pub(super) fn train_qlora(
    config: &LocalTrainingConfig,
    dataset: &TrainingDataset,
    tokenizer: &dyn Tokenizer,
    validation_dataset: Option<&TrainingDataset>,
    bits: u8,
    callback: &dyn Fn(TrainingProgress),
) -> Result<TrainedModelArtifact, TrainingError> {
    let device = WgpuDevice::default();
    let start = Instant::now();
    let rank = config.lora.rank as usize;
    let dim = SafeTensorsLoader::open(&config.model_path)
        .ok()
        .and_then(|loader| loader.load_config())
        .map(|c| c.hidden_size)
        .unwrap_or(rank * 64);

    info!("Initializing QLoRA ({}-bit) training on WGPU device", bits);

    let qlora_config = QLoraLinearConfig::new(dim, dim)
        .with_rank(rank)
        .with_alpha(config.lora.alpha)
        .with_bits(bits);

    let model = if let Some(dequantized) = try_load_quantized_weights(config, dim, bits, &device) {
        qlora_config.init_quantized::<super::types::TrainBackend>(&dequantized, &device)
    } else {
        info!("No quantized weights loaded, using random init for QLoRA");
        qlora_config.init::<super::types::TrainBackend>(&device)
    };

    let batch_size = config.hyperparams.batch_size as usize;
    let steps_per_epoch = dataset.steps_per_epoch(batch_size);
    let total_steps = config.hyperparams.epochs as u64 * steps_per_epoch;

    let lr_schedule = LrSchedule::new(
        config.hyperparams.learning_rate,
        config.hyperparams.warmup_steps,
        total_steps,
        config.hyperparams.lr_scheduler,
    );

    let optim_config = AdamConfig::new().with_weight_decay(Some(
        burn_optim::decay::WeightDecayConfig::new(config.hyperparams.weight_decay as f32),
    ));
    let mut optim = optim_config.init();

    let checkpoint_mgr = CheckpointManager::new(&config.output_dir)
        .with_save_every_steps(500)
        .with_max_checkpoints(3);

    let mut global_step = 0u64;
    let mut model = model;
    let mut running_loss = 0.0f32;

    info!(
        "Training: {} epochs, {} steps/epoch, {} total, lr={}, batch={}",
        config.hyperparams.epochs,
        steps_per_epoch,
        total_steps,
        config.hyperparams.learning_rate,
        batch_size,
    );

    for epoch in 0..config.hyperparams.epochs {
        let epoch_start = Instant::now();

        for step in 0..steps_per_epoch {
            global_step += 1;
            let lr = lr_schedule.get_lr(global_step);

            let batch_start = (step as usize * batch_size) % dataset.len();
            let (input, target) =
                make_batch(dataset, tokenizer, batch_start, batch_size, dim, &device);

            let output = model.forward(input);
            let diff = output - target;
            let loss = diff.clone().powf_scalar(2.0).mean();

            let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap_or_default();
            let loss_scalar = loss_val.first().copied().unwrap_or(0.0);
            running_loss = running_loss * 0.99 + loss_scalar * 0.01;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            if checkpoint_mgr.should_save(global_step) {
                let meta = CheckpointMeta {
                    epoch: epoch + 1,
                    step: global_step,
                    train_loss: running_loss as f64,
                    eval_loss: None,
                    learning_rate: lr,
                    timestamp: chrono::Utc::now(),
                };
                if let Err(e) = checkpoint_mgr.save_meta(global_step, &meta) {
                    warn!("Failed to save checkpoint: {}", e);
                }
            }

            if global_step.is_multiple_of(10) || global_step == total_steps {
                callback(TrainingProgress {
                    epoch: epoch + 1,
                    total_epochs: config.hyperparams.epochs,
                    step: global_step,
                    total_steps,
                    train_loss: Some(running_loss as f64),
                    eval_loss: None,
                    learning_rate: Some(lr),
                    elapsed_secs: start.elapsed().as_secs(),
                });
            }
        }

        // End-of-epoch validation
        let eval_loss = validation_dataset.map(|vd| {
            let vd_steps = vd.steps_per_epoch(batch_size);
            let mut total_loss = 0.0f32;
            for vs in 0..vd_steps {
                let vb_start = (vs as usize * batch_size) % vd.len();
                let (vi, vt) = make_batch(vd, tokenizer, vb_start, batch_size, dim, &device);
                let vo = model.forward(vi);
                let vdiff = vo - vt;
                let vloss = vdiff.clone().powf_scalar(2.0).mean();
                let vl = vloss.into_data().to_vec::<f32>().unwrap_or_default();
                total_loss += vl.first().copied().unwrap_or(0.0);
            }
            let avg = total_loss / vd_steps.max(1) as f32;
            info!(
                "Epoch {}/{} eval_loss: {:.6}",
                epoch + 1,
                config.hyperparams.epochs,
                avg
            );
            avg as f64
        });

        let epoch_duration = epoch_start.elapsed();
        info!(
            "Epoch {}/{} complete in {:.1}s, train_loss: {:.6}{}",
            epoch + 1,
            config.hyperparams.epochs,
            epoch_duration.as_secs_f64(),
            running_loss,
            eval_loss
                .map(|l| format!(", eval_loss: {:.6}", l))
                .unwrap_or_default(),
        );
    }

    let inner = model.valid();
    let a_data = inner.lora_a_weight().into_data();
    let b_data = inner.lora_b_weight().into_data();

    finalize_training(
        config,
        running_loss,
        total_steps,
        &start,
        &a_data.bytes,
        &b_data.bytes,
        None,
    )
}
