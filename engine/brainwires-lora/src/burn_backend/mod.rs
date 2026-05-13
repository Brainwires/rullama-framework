mod alignment;
mod batch;
mod training;
mod types;
mod weights;

pub use types::BurnBackend;

use tracing::info;

use crate::shared::config::{AdapterMethod, AlignmentMethod};
use crate::shared::error::TrainingError;
use crate::dataset_loader::{PreferenceDataset, TrainingDataset};
use crate::{ComputeDevice, LocalTrainingConfig, TrainedModelArtifact, TrainingBackend};
use crate::shared::types::TrainingProgress;

impl TrainingBackend for BurnBackend {
    fn name(&self) -> &str {
        "burn-wgpu"
    }

    fn available_devices(&self) -> Vec<ComputeDevice> {
        let mut devices = vec![ComputeDevice::Cpu];

        // burn-wgpu handles GPU selection internally via WgpuDevice.
        // Report a default GPU device — burn will discover the actual adapter.
        #[cfg(not(target_arch = "wasm32"))]
        devices.push(ComputeDevice::Gpu {
            index: 0,
            name: "Default GPU (WGPU)".to_string(),
            vram_mb: 0,
        });

        devices
    }

    fn train(
        &self,
        config: LocalTrainingConfig,
        callback: Box<dyn Fn(TrainingProgress) + Send>,
    ) -> Result<TrainedModelArtifact, TrainingError> {
        info!("Starting local training with Burn WGPU backend");
        info!("Model: {:?}", config.model_path);
        info!("Dataset: {:?}", config.dataset_path);
        info!("Device: {}", config.device);
        info!(
            "Adapter: {:?}, rank: {}, alpha: {}",
            config.lora.method, config.lora.rank, config.lora.alpha
        );

        if !config.model_path.exists() {
            return Err(TrainingError::Config(format!(
                "Model file not found: {:?}",
                config.model_path
            )));
        }

        if !config.dataset_path.exists() {
            return Err(TrainingError::Config(format!(
                "Dataset file not found: {:?}",
                config.dataset_path
            )));
        }

        std::fs::create_dir_all(&config.output_dir).map_err(|e| {
            TrainingError::Config(format!("Failed to create output directory: {}", e))
        })?;

        // Create tokenizer (BPE if tokenizer_path provided, byte-level otherwise)
        let tokenizer = weights::create_tokenizer(&config)?;

        // Check alignment method first — DPO/ORPO use preference datasets
        match config.alignment {
            AlignmentMethod::DPO { beta } => {
                let pref_dataset = PreferenceDataset::load_jsonl(&config.dataset_path)?;
                info!("Loaded {} preference examples for DPO", pref_dataset.len());
                return alignment::train_dpo_alignment(
                    &config,
                    &pref_dataset,
                    &*tokenizer,
                    beta as f32,
                    &*callback,
                );
            }
            AlignmentMethod::ORPO { lambda } => {
                let pref_dataset = PreferenceDataset::load_jsonl(&config.dataset_path)?;
                info!("Loaded {} preference examples for ORPO", pref_dataset.len());
                return alignment::train_orpo_alignment(
                    &config,
                    &pref_dataset,
                    &*tokenizer,
                    lambda as f32,
                    &*callback,
                );
            }
            AlignmentMethod::None => {}
        }

        // Load SFT dataset
        let dataset = TrainingDataset::load_jsonl(&config.dataset_path)?;
        info!("Loaded {} training examples", dataset.len());

        let validation_dataset = config
            .validation_path
            .as_ref()
            .map(|path| {
                if !path.exists() {
                    return Err(TrainingError::Config(format!(
                        "Validation dataset not found: {:?}",
                        path
                    )));
                }
                TrainingDataset::load_jsonl(path)
            })
            .transpose()?;

        if let Some(ref vd) = validation_dataset {
            info!("Loaded {} validation examples", vd.len());
        }

        // Dispatch based on adapter method
        match config.lora.method {
            AdapterMethod::LoRA => training::train_lora(
                &config,
                &dataset,
                &*tokenizer,
                validation_dataset.as_ref(),
                &*callback,
            ),
            AdapterMethod::DoRA => training::train_dora(
                &config,
                &dataset,
                &*tokenizer,
                validation_dataset.as_ref(),
                &*callback,
            ),
            AdapterMethod::QLoRA { bits } => training::train_qlora(
                &config,
                &dataset,
                &*tokenizer,
                validation_dataset.as_ref(),
                bits,
                &*callback,
            ),
            AdapterMethod::QDoRA { bits } => {
                info!(
                    "QDoRA ({}-bit): using DoRA training path with quantized weights",
                    bits
                );
                training::train_dora(
                    &config,
                    &dataset,
                    &*tokenizer,
                    validation_dataset.as_ref(),
                    &*callback,
                )
            }
        }
    }
}
