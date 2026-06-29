//! Docker-based sandboxed code execution backend.
//!
//! Provides a `DockerExecutor` that shells out to the Docker CLI to run
//! untrusted code inside isolated containers with configurable resource
//! limits, network restrictions, and read-only filesystems.
//!
//! # Example
//!
//! ```rust,no_run
//! use rullama_tool_builtins::interpreters::docker::{DockerExecutor, DockerConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let executor = DockerExecutor::new(DockerConfig::default());
//!
//! if DockerExecutor::is_available().await {
//!     executor.pull_image().await?;
//!     let result = executor.execute("python", "print('hello')").await?;
//!     println!("stdout: {}", result.stdout);
//! }
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use tokio::process::Command;

/// Configuration for the Docker execution backend.
#[derive(Debug, Clone)]
pub struct DockerConfig {
    /// Docker image to use (default: `"rullama/sandbox:latest"`).
    pub image: String,
    /// Memory limit in bytes (default: 256 MB).
    pub memory_limit: u64,
    /// CPU limit in cores (default: 1.0).
    pub cpu_limit: f64,
    /// Disable network access (default: `true`).
    pub network_disabled: bool,
    /// Execution timeout (default: 30 s).
    pub timeout: Duration,
    /// Mount a working directory into the container (default: `false`).
    pub mount_workdir: bool,
    /// Optional working directory path on the host.
    pub workdir: Option<PathBuf>,
}

impl Default for DockerConfig {
    fn default() -> Self {
        Self {
            image: "rullama/sandbox:latest".to_string(),
            memory_limit: 256 * 1024 * 1024, // 256 MB
            cpu_limit: 1.0,
            network_disabled: true,
            timeout: Duration::from_secs(30),
            mount_workdir: false,
            workdir: None,
        }
    }
}

/// Result of a Docker-based code execution.
#[derive(Debug, Clone)]
pub struct DockerExecutionResult {
    /// Captured standard output.
    pub stdout: String,
    /// Captured standard error.
    pub stderr: String,
    /// Container process exit code.
    pub exit_code: i32,
    /// Wall-clock duration of the execution.
    pub duration: Duration,
    /// Whether the execution was killed due to timeout.
    pub timed_out: bool,
}

/// Docker-backed code executor.
///
/// Builds and invokes `docker run` commands with security-hardened flags.
pub struct DockerExecutor {
    config: DockerConfig,
}

impl DockerExecutor {
    /// Create a new executor with the given configuration.
    pub fn new(config: DockerConfig) -> Self {
        Self { config }
    }

    /// Execute `code` written in `language` inside a Docker container.
    ///
    /// The source is written to a temporary file, mounted read-only into
    /// the container, and executed with the appropriate interpreter.
    pub async fn execute(
        &self,
        language: &str,
        code: &str,
    ) -> anyhow::Result<DockerExecutionResult> {
        // Determine the file extension for the language.
        let ext = language_extension(language)?;

        // Write code to a temporary file.
        let tmp_dir = tempfile::tempdir()?;
        let code_filename = format!("code.{ext}");
        let host_path = tmp_dir.path().join(&code_filename);
        std::fs::write(&host_path, code)?;

        let container_path = format!("/sandbox/{code_filename}");

        // Build docker command arguments.
        let args = self.build_docker_args(&host_path, &container_path, language)?;

        let start = Instant::now();

        let result = tokio::time::timeout(self.config.timeout, async {
            Command::new("docker").args(&args).output().await
        })
        .await;

        let duration = start.elapsed();

        match result {
            Ok(Ok(output)) => Ok(DockerExecutionResult {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
                duration,
                timed_out: false,
            }),
            Ok(Err(e)) => Err(anyhow::anyhow!("Failed to run docker: {e}")),
            Err(_) => {
                // Timeout — the container may still be running.  The `--rm`
                // flag ensures it is cleaned up once the process exits, and
                // Docker's own OOM-killer will eventually reap it.
                Ok(DockerExecutionResult {
                    stdout: String::new(),
                    stderr: "Execution timed out".to_string(),
                    exit_code: -1,
                    duration,
                    timed_out: true,
                })
            }
        }
    }

    /// Check whether the Docker CLI is accessible on this machine.
    pub async fn is_available() -> bool {
        Command::new("docker")
            .arg("--version")
            .output()
            .await
            .is_ok_and(|o| o.status.success())
    }

    /// Pull the configured sandbox image if not already present.
    pub async fn pull_image(&self) -> anyhow::Result<()> {
        let output = Command::new("docker")
            .args(["pull", &self.config.image])
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("docker pull failed: {stderr}");
        }
        Ok(())
    }

    /// List Docker images available on this host.
    pub async fn list_images() -> anyhow::Result<Vec<String>> {
        let output = Command::new("docker")
            .args(["images", "--format", "{{.Repository}}:{{.Tag}}"])
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("docker images failed: {stderr}");
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.lines().map(String::from).collect())
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Build the argument list for `docker run`.
    ///
    /// This is deliberately `pub(crate)` so that unit tests can inspect the
    /// generated command without actually invoking Docker.
    pub(crate) fn build_docker_args(
        &self,
        host_code_path: &std::path::Path,
        container_code_path: &str,
        language: &str,
    ) -> anyhow::Result<Vec<String>> {
        let mut args: Vec<String> = vec!["run".into(), "--rm".into()];

        // Resource limits.
        args.push("--memory".into());
        args.push(self.config.memory_limit.to_string());
        args.push("--cpus".into());
        args.push(format!("{:.1}", self.config.cpu_limit));

        // Network isolation.
        if self.config.network_disabled {
            args.push("--network".into());
            args.push("none".into());
        }

        // Read-only root filesystem.
        args.push("--read-only".into());

        // Mount the code file read-only.
        args.push("-v".into());
        args.push(format!(
            "{}:{}:ro",
            host_code_path.display(),
            container_code_path
        ));

        // Optionally mount a working directory.
        if self.config.mount_workdir {
            if let Some(ref workdir) = self.config.workdir {
                args.push("-v".into());
                args.push(format!("{}:/sandbox/work", workdir.display()));
            }
        }

        // Provide a writable /tmp inside the read-only filesystem.
        args.push("--tmpfs".into());
        args.push("/tmp:rw,noexec,nosuid,size=64m".into());

        // Image name.
        args.push(self.config.image.clone());

        // Interpreter command.
        let cmd = language_command(language, container_code_path)?;
        args.extend(cmd);

        Ok(args)
    }
}

/// Map a language identifier to its file extension.
fn language_extension(language: &str) -> anyhow::Result<&'static str> {
    match language {
        "python" | "py" => Ok("py"),
        "javascript" | "js" | "node" => Ok("js"),
        "lua" => Ok("lua"),
        "ruby" | "rb" => Ok("rb"),
        "bash" | "sh" => Ok("sh"),
        _ => Err(anyhow::anyhow!(
            "Unsupported language for Docker execution: {language}"
        )),
    }
}

/// Map a language identifier to the command used to run the code inside the
/// container.
fn language_command(language: &str, file_path: &str) -> anyhow::Result<Vec<String>> {
    match language {
        "python" | "py" => Ok(vec!["python3".into(), file_path.into()]),
        "javascript" | "js" | "node" => Ok(vec!["node".into(), file_path.into()]),
        "lua" => Ok(vec!["lua".into(), file_path.into()]),
        "ruby" | "rb" => Ok(vec!["ruby".into(), file_path.into()]),
        "bash" | "sh" => Ok(vec!["bash".into(), file_path.into()]),
        _ => Err(anyhow::anyhow!(
            "Unsupported language for Docker execution: {language}"
        )),
    }
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = DockerConfig::default();
        assert_eq!(cfg.image, "rullama/sandbox:latest");
        assert_eq!(cfg.memory_limit, 256 * 1024 * 1024);
        assert!((cfg.cpu_limit - 1.0).abs() < f64::EPSILON);
        assert!(cfg.network_disabled);
        assert_eq!(cfg.timeout, Duration::from_secs(30));
        assert!(!cfg.mount_workdir);
        assert!(cfg.workdir.is_none());
    }

    #[test]
    fn test_language_command_python() {
        let cmd = language_command("python", "/sandbox/code.py").unwrap();
        assert_eq!(cmd, vec!["python3", "/sandbox/code.py"]);
    }

    #[test]
    fn test_language_command_py_alias() {
        let cmd = language_command("py", "/sandbox/code.py").unwrap();
        assert_eq!(cmd, vec!["python3", "/sandbox/code.py"]);
    }

    #[test]
    fn test_language_command_javascript() {
        let cmd = language_command("javascript", "/sandbox/code.js").unwrap();
        assert_eq!(cmd, vec!["node", "/sandbox/code.js"]);
    }

    #[test]
    fn test_language_command_js_alias() {
        let cmd = language_command("js", "/sandbox/code.js").unwrap();
        assert_eq!(cmd, vec!["node", "/sandbox/code.js"]);
    }

    #[test]
    fn test_language_command_node_alias() {
        let cmd = language_command("node", "/sandbox/code.js").unwrap();
        assert_eq!(cmd, vec!["node", "/sandbox/code.js"]);
    }

    #[test]
    fn test_language_command_lua() {
        let cmd = language_command("lua", "/sandbox/code.lua").unwrap();
        assert_eq!(cmd, vec!["lua", "/sandbox/code.lua"]);
    }

    #[test]
    fn test_language_command_ruby() {
        let cmd = language_command("ruby", "/sandbox/code.rb").unwrap();
        assert_eq!(cmd, vec!["ruby", "/sandbox/code.rb"]);
    }

    #[test]
    fn test_language_command_bash() {
        let cmd = language_command("bash", "/sandbox/code.sh").unwrap();
        assert_eq!(cmd, vec!["bash", "/sandbox/code.sh"]);
    }

    #[test]
    fn test_language_command_sh_alias() {
        let cmd = language_command("sh", "/sandbox/code.sh").unwrap();
        assert_eq!(cmd, vec!["bash", "/sandbox/code.sh"]);
    }

    #[test]
    fn test_language_command_unsupported() {
        let result = language_command("cobol", "/sandbox/code.cob");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unsupported language"));
        assert!(err.contains("cobol"));
    }

    #[test]
    fn test_language_extension() {
        assert_eq!(language_extension("python").unwrap(), "py");
        assert_eq!(language_extension("py").unwrap(), "py");
        assert_eq!(language_extension("javascript").unwrap(), "js");
        assert_eq!(language_extension("js").unwrap(), "js");
        assert_eq!(language_extension("lua").unwrap(), "lua");
        assert_eq!(language_extension("ruby").unwrap(), "rb");
        assert_eq!(language_extension("bash").unwrap(), "sh");
        assert!(language_extension("fortran").is_err());
    }

    #[test]
    fn test_build_docker_args_default_config() {
        let executor = DockerExecutor::new(DockerConfig::default());
        let host_path = std::path::Path::new("/tmp/code.py");
        let container_path = "/sandbox/code.py";

        let args = executor
            .build_docker_args(host_path, container_path, "python")
            .unwrap();

        // Must start with "run --rm"
        assert_eq!(args[0], "run");
        assert_eq!(args[1], "--rm");

        // Must contain memory limit
        let mem_idx = args.iter().position(|a| a == "--memory").unwrap();
        assert_eq!(args[mem_idx + 1], (256u64 * 1024 * 1024).to_string());

        // Must contain cpu limit
        let cpu_idx = args.iter().position(|a| a == "--cpus").unwrap();
        assert_eq!(args[cpu_idx + 1], "1.0");

        // Must contain network none (default)
        let net_idx = args.iter().position(|a| a == "--network").unwrap();
        assert_eq!(args[net_idx + 1], "none");

        // Must contain --read-only
        assert!(args.contains(&"--read-only".to_string()));

        // Must mount code file read-only
        let vol_idx = args.iter().position(|a| a == "-v").unwrap();
        assert_eq!(args[vol_idx + 1], "/tmp/code.py:/sandbox/code.py:ro");

        // Must contain --tmpfs for writable /tmp
        let tmpfs_idx = args.iter().position(|a| a == "--tmpfs").unwrap();
        assert!(args[tmpfs_idx + 1].starts_with("/tmp:"));

        // Must contain image name
        assert!(args.contains(&"rullama/sandbox:latest".to_string()));

        // Must end with the interpreter command
        let last_two = &args[args.len() - 2..];
        assert_eq!(last_two, &["python3", "/sandbox/code.py"]);
    }

    #[test]
    fn test_build_docker_args_network_enabled() {
        let config = DockerConfig {
            network_disabled: false,
            ..DockerConfig::default()
        };
        let executor = DockerExecutor::new(config);
        let host_path = std::path::Path::new("/tmp/code.js");

        let args = executor
            .build_docker_args(host_path, "/sandbox/code.js", "js")
            .unwrap();

        // Should NOT contain --network none
        assert!(!args.contains(&"--network".to_string()));
    }

    #[test]
    fn test_build_docker_args_with_workdir() {
        let config = DockerConfig {
            mount_workdir: true,
            workdir: Some(PathBuf::from("/home/user/project")),
            ..DockerConfig::default()
        };
        let executor = DockerExecutor::new(config);
        let host_path = std::path::Path::new("/tmp/code.lua");

        let args = executor
            .build_docker_args(host_path, "/sandbox/code.lua", "lua")
            .unwrap();

        // Should have two -v mounts: the code file and the workdir
        let vol_indices: Vec<_> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == "-v")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(vol_indices.len(), 2);

        // Second mount should be the workdir
        assert_eq!(args[vol_indices[1] + 1], "/home/user/project:/sandbox/work");
    }

    #[test]
    fn test_build_docker_args_custom_image() {
        let config = DockerConfig {
            image: "my-custom-image:v2".to_string(),
            ..DockerConfig::default()
        };
        let executor = DockerExecutor::new(config);
        let host_path = std::path::Path::new("/tmp/code.rb");

        let args = executor
            .build_docker_args(host_path, "/sandbox/code.rb", "ruby")
            .unwrap();

        assert!(args.contains(&"my-custom-image:v2".to_string()));
        // Last two args should be the ruby command
        let last_two = &args[args.len() - 2..];
        assert_eq!(last_two, &["ruby", "/sandbox/code.rb"]);
    }

    #[test]
    fn test_build_docker_args_unsupported_language() {
        let executor = DockerExecutor::new(DockerConfig::default());
        let host_path = std::path::Path::new("/tmp/code.cob");

        let result = executor.build_docker_args(host_path, "/sandbox/code.cob", "cobol");
        assert!(result.is_err());
    }
}
