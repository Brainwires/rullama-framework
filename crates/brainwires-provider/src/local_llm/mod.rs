//! Local LLM Provider Module
//!
//! This module provides support for running local LLM models using llama.cpp.
//! It's designed for CPU-optimized inference with models like:
//! - LFM2 (Liquid Foundation Model 2) - Hybrid architecture with excellent CPU performance
//! - Granite 4.0 Nano - Transformer architecture optimized for CPU
//!
//! Key features:
//! - GGUF model format support
//! - CPU-optimized inference (no GPU required)
//! - Memory-efficient with small models (100MB - 2GB RAM)
//! - 32K context window support
//! - Streaming text generation

mod config;
mod model_registry;
mod provider;

#[cfg(not(target_arch = "wasm32"))]
pub mod ollama_cache;

pub use config::*;
pub use model_registry::*;
pub use provider::*;
