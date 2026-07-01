use anyhow::Result;
use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Trait for server-side MCP transport.
#[async_trait]
pub trait ServerTransport: Send + Sync {
    /// Read the next JSON-RPC request, or None on EOF.
    async fn read_request(&mut self) -> Result<Option<String>>;
    /// Write a JSON-RPC response.
    async fn write_response(&mut self, response: &str) -> Result<()>;
}

/// Stdio-based server transport (stdin/stdout).
pub struct StdioServerTransport {
    reader: BufReader<tokio::io::Stdin>,
    stdout: tokio::io::Stdout,
}

impl StdioServerTransport {
    /// Create a new stdio transport.
    pub fn new() -> Self {
        Self {
            reader: BufReader::new(tokio::io::stdin()),
            stdout: tokio::io::stdout(),
        }
    }
}

impl Default for StdioServerTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ServerTransport for StdioServerTransport {
    async fn read_request(&mut self) -> Result<Option<String>> {
        let mut line = String::new();
        let bytes_read = self.reader.read_line(&mut line).await?;
        if bytes_read == 0 {
            return Ok(None); // EOF
        }
        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            return Ok(None);
        }
        Ok(Some(trimmed))
    }

    async fn write_response(&mut self, response: &str) -> Result<()> {
        self.stdout
            .write_all(format!("{response}\n").as_bytes())
            .await?;
        self.stdout.flush().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdio_transport_default() {
        // Just ensure it can be constructed without panic
        let _transport = StdioServerTransport::new();
    }
}
