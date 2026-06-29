//! Example: Implementing a custom embedding provider
//!
//! Shows how to implement the `EmbeddingProvider` trait to plug in any
//! embedding model (sentence-transformers, OpenAI embeddings, custom model, etc.).
//!
//! Run: cargo run -p rullama --example custom_embedding

use anyhow::Result;
use rullama::prelude::*;

/// A toy embedding provider that produces deterministic vectors.
/// Replace with your real embedding model (e.g., ONNX Runtime, HTTP API call).
struct HashEmbedding {
    dim: usize,
}

impl HashEmbedding {
    fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl EmbeddingProvider for HashEmbedding {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Simple hash-based embedding (deterministic, not semantic)
        let mut vec = vec![0.0f32; self.dim];
        for (i, byte) in text.bytes().enumerate() {
            vec[i % self.dim] += byte as f32 / 255.0;
        }
        // Normalize to unit length
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        Ok(vec)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        "hash-embedding-v1"
    }

    // embed_batch has a default impl that calls embed() in a loop.
    // Override it if your backend supports native batching for better throughput.
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() -> Result<()> {
    let provider = HashEmbedding::new(64);

    println!("Model: {}", provider.model_name());
    println!("Dimension: {}", provider.dimension());

    // Embed some texts
    let a = provider.embed("machine learning")?;
    let b = provider.embed("deep learning")?;
    let c = provider.embed("banana smoothie")?;

    println!("\nCosine similarities:");
    println!(
        "  'machine learning' vs 'deep learning': {:.4}",
        cosine_similarity(&a, &b)
    );
    println!(
        "  'machine learning' vs 'banana smoothie': {:.4}",
        cosine_similarity(&a, &c)
    );

    // Batch embedding
    let texts = vec!["first document".to_string(), "second document".to_string()];
    let batch = provider.embed_batch(&texts)?;
    println!(
        "\nBatch embedded {} texts, each with {} dimensions",
        batch.len(),
        batch[0].len()
    );

    Ok(())
}
