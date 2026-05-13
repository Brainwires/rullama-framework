use std::time::Instant;

use burn_core::module::AutodiffModule;
use burn_core::prelude::*;
use burn_optim::{AdamConfig, GradientsParams, Optimizer};
use burn_wgpu::WgpuDevice;
use tracing::info;

use super::batch::make_preference_batch;
use super::weights::{finalize_training, try_load_safetensors_weights};
use crate::shared::error::TrainingError;
use crate::burn_modules::{LoraLinearConfig, dpo_loss, orpo_loss};
use crate::dataset_loader::{PreferenceDataset, Tokenizer};
use crate::lr_schedule::LrSchedule;
use crate::weight_loader::SafeTensorsLoader;
use crate::{LocalTrainingConfig, TrainedModelArtifact};
use crate::shared::types::TrainingProgress;

use super::types::TrainBackend;

/// Run DPO alignment training with preference pairs.
pub(super) fn train_dpo_alignment(
    config: &LocalTrainingConfig,
    pref_dataset: &PreferenceDataset,
    tokenizer: &dyn Tokenizer,
    beta: f32,
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

    info!(
        "Initializing DPO alignment training (beta={}) on WGPU device",
        beta
    );

    let lora_config = LoraLinearConfig::new(dim, dim)
        .with_rank(rank)
        .with_alpha(config.lora.alpha);

    let model = if let Some(base_weight) = try_load_safetensors_weights(config, dim, &device) {
        lora_config.init_with_base_weights::<TrainBackend>(base_weight, &device)
    } else {
        lora_config.init::<TrainBackend>(&device)
    };

    // Clone initial adapter weights as frozen reference model
    let ref_model = model.valid();

    let batch_size = config.hyperparams.batch_size as usize;
    let steps_per_epoch = pref_dataset.steps_per_epoch(batch_size);
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

    let mut global_step = 0u64;
    let mut model = model;
    let mut running_loss = 0.0f32;

    info!(
        "DPO Training: {} epochs, {} steps/epoch, {} total, beta={}",
        config.hyperparams.epochs, steps_per_epoch, total_steps, beta,
    );

    for epoch in 0..config.hyperparams.epochs {
        for step in 0..steps_per_epoch {
            global_step += 1;
            let lr = lr_schedule.get_lr(global_step);

            let batch_start = (step as usize * batch_size) % pref_dataset.len();
            let (input, chosen, rejected) = make_preference_batch(
                pref_dataset,
                tokenizer,
                batch_start,
                batch_size,
                dim,
                &device,
            );

            // Policy model: forward chosen and rejected
            let policy_chosen_out = model.forward(input.clone() + chosen.clone());
            let policy_rejected_out = model.forward(input.clone() + rejected.clone());

            // Compute per-sample "log-probs" as negative MSE (proxy for actual log-probs)
            let policy_chosen_logps = (policy_chosen_out - chosen.clone())
                .powf_scalar(2.0)
                .mean_dim(1)
                .neg()
                .squeeze::<1>();
            let policy_rejected_logps = (policy_rejected_out - rejected.clone())
                .powf_scalar(2.0)
                .mean_dim(1)
                .neg()
                .squeeze::<1>();

            // Reference model: same but no gradient (uses inner backend)
            let ref_input_chosen = (input.clone() + chosen.clone()).inner();
            let ref_input_rejected = (input + rejected.clone()).inner();
            let chosen_inner = chosen.inner();
            let rejected_inner = rejected.inner();

            let ref_chosen_out = ref_model.forward(ref_input_chosen);
            let ref_rejected_out = ref_model.forward(ref_input_rejected);

            let ref_chosen_logps_inner = (ref_chosen_out - chosen_inner)
                .powf_scalar(2.0)
                .mean_dim(1)
                .neg()
                .squeeze::<1>();
            let ref_rejected_logps_inner = (ref_rejected_out - rejected_inner)
                .powf_scalar(2.0)
                .mean_dim(1)
                .neg()
                .squeeze::<1>();

            // Wrap back into autodiff tensors (as constants, no grad)
            let ref_chosen_logps = Tensor::<TrainBackend, 1>::from_inner(ref_chosen_logps_inner);
            let ref_rejected_logps =
                Tensor::<TrainBackend, 1>::from_inner(ref_rejected_logps_inner);

            let loss = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta,
            );

            let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap_or_default();
            let loss_scalar = loss_val.first().copied().unwrap_or(0.0);
            running_loss = running_loss * 0.99 + loss_scalar * 0.01;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

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

        info!(
            "DPO Epoch {}/{} complete, loss: {:.6}",
            epoch + 1,
            config.hyperparams.epochs,
            running_loss,
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

/// Run ORPO alignment training with preference pairs.
pub(super) fn train_orpo_alignment(
    config: &LocalTrainingConfig,
    pref_dataset: &PreferenceDataset,
    tokenizer: &dyn Tokenizer,
    lambda: f32,
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

    info!(
        "Initializing ORPO alignment training (lambda={}) on WGPU device",
        lambda
    );

    let lora_config = LoraLinearConfig::new(dim, dim)
        .with_rank(rank)
        .with_alpha(config.lora.alpha);

    let model = if let Some(base_weight) = try_load_safetensors_weights(config, dim, &device) {
        lora_config.init_with_base_weights::<TrainBackend>(base_weight, &device)
    } else {
        lora_config.init::<TrainBackend>(&device)
    };

    let batch_size = config.hyperparams.batch_size as usize;
    let steps_per_epoch = pref_dataset.steps_per_epoch(batch_size);
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

    let mut global_step = 0u64;
    let mut model = model;
    let mut running_loss = 0.0f32;

    info!(
        "ORPO Training: {} epochs, {} steps/epoch, {} total, lambda={}",
        config.hyperparams.epochs, steps_per_epoch, total_steps, lambda,
    );

    for epoch in 0..config.hyperparams.epochs {
        for step in 0..steps_per_epoch {
            global_step += 1;
            let lr = lr_schedule.get_lr(global_step);

            let batch_start = (step as usize * batch_size) % pref_dataset.len();
            let (input, chosen, rejected) = make_preference_batch(
                pref_dataset,
                tokenizer,
                batch_start,
                batch_size,
                dim,
                &device,
            );

            // Forward through model for chosen and rejected
            let chosen_out = model.forward(input.clone() + chosen.clone());
            let rejected_out = model.forward(input.clone() + rejected.clone());

            // SFT loss on chosen completions
            let sft_diff = chosen_out.clone() - chosen.clone();
            let sft_loss = sft_diff.powf_scalar(2.0).mean();

            // Compute "probabilities" as softmax of negative MSE per sample
            let chosen_scores = (chosen_out - chosen)
                .powf_scalar(2.0)
                .mean_dim(1)
                .neg()
                .squeeze::<1>();
            let rejected_scores = (rejected_out - rejected)
                .powf_scalar(2.0)
                .mean_dim(1)
                .neg()
                .squeeze::<1>();

            // Convert to probabilities via sigmoid
            let chosen_probs = burn_core::tensor::activation::sigmoid(chosen_scores);
            let rejected_probs = burn_core::tensor::activation::sigmoid(rejected_scores);

            let loss = orpo_loss(sft_loss, chosen_probs, rejected_probs, lambda);

            let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap_or_default();
            let loss_scalar = loss_val.first().copied().unwrap_or(0.0);
            running_loss = running_loss * 0.99 + loss_scalar * 0.01;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

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

        info!(
            "ORPO Epoch {}/{} complete, loss: {:.6}",
            epoch + 1,
            config.hyperparams.epochs,
            running_loss,
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
