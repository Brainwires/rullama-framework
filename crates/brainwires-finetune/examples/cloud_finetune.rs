//! Example: Cloud fine-tuning with the TrainingManager
//!
//! Demonstrates how to configure a cloud fine-tuning job with provider
//! selection, hyperparameters, LoRA settings, and job status checking.
//!
//! Run: cargo run -p brainwires-finetune --example cloud_finetune --features cloud

use brainwires_finetune::{
    AlignmentMethod, CloudFineTuneConfig, DatasetId, FineTuneProviderFactory, LrScheduler,
    TrainingHyperparams, TrainingJobStatus, TrainingManager,
};

#[tokio::main]
async fn main() {
    // Step 1: Create a TrainingManager and register a cloud provider.
    // In production you would use a real API key from an environment variable.
    let mut manager = TrainingManager::new();
    let openai = FineTuneProviderFactory::openai("sk-demo-not-a-real-key");
    manager.add_cloud_provider(Box::new(openai));

    // You can register multiple providers at once.
    let together = FineTuneProviderFactory::together("tog-demo-key");
    manager.add_cloud_provider(Box::new(together));

    println!("Registered providers: {:?}", manager.cloud_providers());

    // Step 2: Configure training hyperparameters.
    let hyperparams = TrainingHyperparams {
        epochs: 2,
        batch_size: 8,
        learning_rate: 5e-5,
        warmup_steps: 50,
        weight_decay: 0.01,
        lr_scheduler: LrScheduler::Cosine,
        seed: 42,
        max_seq_len: 4096,
        gradient_accumulation_steps: 2,
        max_grad_norm: 1.0,
    };

    // Step 3: Build the fine-tune configuration.
    // DatasetId wraps the provider-specific dataset identifier you receive
    // after uploading your JSONL training data.
    let config = CloudFineTuneConfig::new("gpt-4o-mini-2024-07-18", DatasetId::from("file-abc123"))
        .with_validation(DatasetId::from("file-val456"))
        .with_hyperparams(hyperparams)
        .with_alignment(AlignmentMethod::dpo())
        .with_suffix("customer-support-v1");

    println!("Fine-tune config: {:#?}", config);

    // Step 4: Submit the job (skipped here since we have no real API key).
    // In production:
    //
    //   let job_id = manager.start_cloud_job("openai", config).await.unwrap();
    //   println!("Started job: {}", job_id);

    // Step 5: Check job status.
    // Simulate a job ID that would be returned by the provider.
    let demo_job_id: brainwires_finetune::TrainingJobId = "ftjob-demo-000".into();

    // In production you would call:
    //
    //   let status = manager.check_cloud_job("openai", &demo_job_id).await.unwrap();
    //
    // Or wait for completion with exponential-backoff polling:
    //
    //   let final_status = manager.wait_for_cloud_job("openai", &demo_job_id).await.unwrap();

    // Step 6: Interpret the final status.
    // Here we demonstrate pattern matching on the status enum.
    let simulated_status = TrainingJobStatus::Succeeded {
        model_id: "ft:gpt-4o-mini:my-org:customer-support-v1:9abc123".to_string(),
    };

    match &simulated_status {
        TrainingJobStatus::Succeeded { model_id } => {
            println!("Training complete! Fine-tuned model: {}", model_id);
        }
        TrainingJobStatus::Failed { error } => {
            eprintln!("Training failed: {}", error);
        }
        TrainingJobStatus::Running { progress } => {
            println!(
                "Training in progress: {:.1}% (epoch {}/{})",
                progress.completion_fraction() * 100.0,
                progress.epoch,
                progress.total_epochs,
            );
        }
        other => {
            println!("Job {} status: {:?}", demo_job_id, other);
        }
    }
}
