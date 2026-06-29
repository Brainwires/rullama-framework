/// Startup configuration parsed from CLI flags.
pub struct Config {
    /// Directory where `jobs.json` and per-job logs are stored.
    pub jobs_dir: std::path::PathBuf,
    /// Maximum number of jobs that may run concurrently.
    pub max_concurrent: usize,
    /// Optional HTTP listen address.  `None` means stdio-only mode.
    pub http_addr: Option<String>,
}

impl Config {
    /// Resolve the jobs directory: `--jobs-dir` flag → `~/.brainwires/scheduler/`.
    pub fn resolve_jobs_dir(flag: Option<String>) -> std::path::PathBuf {
        if let Some(p) = flag {
            return std::path::PathBuf::from(p);
        }
        dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".brainwires")
            .join("scheduler")
    }
}
