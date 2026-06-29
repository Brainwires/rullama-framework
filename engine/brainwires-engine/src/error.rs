use thiserror::Error;

#[derive(Debug, Error)]
pub enum RullamaError {
    #[error("WebGPU is not available in this environment")]
    WebGpuUnavailable,

    #[error("failed to request a wgpu adapter")]
    NoAdapter,

    #[error("failed to request a wgpu device: {0}")]
    DeviceRequest(String),

    #[error("GGUF parse error: {0}")]
    Gguf(String),

    #[error("image model error: {0}")]
    Image(String),

    #[error("model config error: {0}")]
    Config(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("buffer mapping failed: {0}")]
    BufferMap(String),

    #[error("cancelled by caller")]
    Cancelled,
}

pub type Result<T, E = RullamaError> = core::result::Result<T, E>;
