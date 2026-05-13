use burn_core::prelude::*;
use burn_wgpu::WgpuDevice;

use super::types::TrainBackend;
use crate::dataset_loader::{PreferenceDataset, Tokenizer, TrainingDataset};

/// Build a training batch from dataset examples as (input, target) tensors.
pub(super) fn make_batch(
    dataset: &TrainingDataset,
    tokenizer: &dyn Tokenizer,
    batch_start: usize,
    batch_size: usize,
    dim: usize,
    device: &WgpuDevice,
) -> (Tensor<TrainBackend, 2>, Tensor<TrainBackend, 2>) {
    let batch = dataset.get_batch(batch_start, batch_size);
    let actual_batch = batch.len().max(1);

    let mut input_data = vec![0.0f32; actual_batch * dim];
    let mut target_data = vec![0.0f32; actual_batch * dim];

    for (i, example) in batch.iter().enumerate() {
        let (input_ids, target_ids) = tokenizer.encode_example(example);
        for (j, &tok) in input_ids.iter().take(dim).enumerate() {
            input_data[i * dim + j] = (tok as f32 / 128.0) - 1.0;
        }
        for (j, &tok) in target_ids.iter().take(dim).enumerate() {
            if tok != u32::MAX {
                target_data[i * dim + j] = (tok as f32 / 128.0) - 1.0;
            }
        }
    }

    let input = Tensor::from_floats(
        burn_core::tensor::TensorData::new(input_data, [actual_batch, dim]),
        device,
    );
    let target = Tensor::from_floats(
        burn_core::tensor::TensorData::new(target_data, [actual_batch, dim]),
        device,
    );

    (input, target)
}

/// Build a preference batch as (prompt_input, chosen_target, rejected_target) tensors.
pub(super) fn make_preference_batch(
    dataset: &PreferenceDataset,
    tokenizer: &dyn Tokenizer,
    batch_start: usize,
    batch_size: usize,
    dim: usize,
    device: &WgpuDevice,
) -> (
    Tensor<TrainBackend, 2>,
    Tensor<TrainBackend, 2>,
    Tensor<TrainBackend, 2>,
) {
    let batch = dataset.get_batch(batch_start, batch_size);
    let actual_batch = batch.len().max(1);

    let mut input_data = vec![0.0f32; actual_batch * dim];
    let mut chosen_data = vec![0.0f32; actual_batch * dim];
    let mut rejected_data = vec![0.0f32; actual_batch * dim];

    for (i, example) in batch.iter().enumerate() {
        let prompt_tokens = tokenizer.encode(&example.prompt);
        let chosen_tokens = tokenizer.encode(&example.chosen);
        let rejected_tokens = tokenizer.encode(&example.rejected);

        for (j, &tok) in prompt_tokens.iter().take(dim).enumerate() {
            input_data[i * dim + j] = (tok as f32 / 128.0) - 1.0;
        }
        for (j, &tok) in chosen_tokens.iter().take(dim).enumerate() {
            chosen_data[i * dim + j] = (tok as f32 / 128.0) - 1.0;
        }
        for (j, &tok) in rejected_tokens.iter().take(dim).enumerate() {
            rejected_data[i * dim + j] = (tok as f32 / 128.0) - 1.0;
        }
    }

    let input = Tensor::from_floats(
        burn_core::tensor::TensorData::new(input_data, [actual_batch, dim]),
        device,
    );
    let chosen = Tensor::from_floats(
        burn_core::tensor::TensorData::new(chosen_data, [actual_batch, dim]),
        device,
    );
    let rejected = Tensor::from_floats(
        burn_core::tensor::TensorData::new(rejected_data, [actual_batch, dim]),
        device,
    );

    (input, chosen, rejected)
}
