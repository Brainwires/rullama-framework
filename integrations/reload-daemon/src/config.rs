use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct DaemonConfig {
    pub listen: String,
    pub clients: HashMap<String, ClientStrategy>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClientStrategy {
    pub process_name: String,
    pub kill_signals: Vec<String>,
    pub kill_timeouts_ms: Vec<u64>,
    pub restart_args_transform: Option<ArgsTransform>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ArgsTransform {
    pub preserve_flags: Vec<String>,
    pub replace_trailing: Vec<String>,
}
