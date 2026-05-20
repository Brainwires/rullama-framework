//! File system watcher using the `notify` crate.

use std::path::PathBuf;
use std::sync::Arc;

use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::{RwLock, mpsc, watch};

use super::debounce::EventDebouncer;
use super::rules::{FsEventType, ReactorRule};
use crate::system::config::ReactorConfig;

/// File system reactor that watches directories and dispatches actions.
///
/// Registers file system watchers for all enabled rules, processes events
/// through the debouncer, and sends matching (rule, path, event) tuples
/// to an action channel.
pub struct FsReactor {
    config: ReactorConfig,
    rules: Vec<ReactorRule>,
}

impl FsReactor {
    /// Create a new file system reactor.
    pub fn new(config: ReactorConfig, rules: Vec<ReactorRule>) -> Self {
        Self { config, rules }
    }

    /// Run the reactor, watching for file system events until cancelled.
    pub async fn run(
        &self,
        action_tx: mpsc::Sender<(ReactorRule, String, FsEventType)>,
        mut cancel: watch::Receiver<bool>,
    ) -> anyhow::Result<()> {
        let (fs_tx, mut fs_rx) = mpsc::channel::<Event>(256);
        let debouncer = Arc::new(RwLock::new(EventDebouncer::new(
            self.config.global_debounce_ms,
            self.config.max_events_per_minute,
        )));

        // Create the watcher
        let tx_clone = fs_tx.clone();
        let mut watcher = RecommendedWatcher::new(
            move |res: Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    let _ = tx_clone.blocking_send(event);
                }
            },
            Config::default(),
        )?;

        // Register watch paths from all enabled rules
        let mut watched_paths = std::collections::HashSet::new();
        for rule in &self.rules {
            if !rule.enabled {
                continue;
            }
            for path in &rule.watch_paths {
                if watched_paths.insert(path.clone()) {
                    let watch_path = PathBuf::from(path);
                    if watch_path.exists() {
                        watcher.watch(&watch_path, RecursiveMode::Recursive)?;
                        tracing::info!("Watching: {path}");
                    } else {
                        tracing::warn!("Watch path does not exist: {path}");
                    }
                }
            }
        }

        tracing::info!(
            "File system reactor started with {} rules, watching {} paths",
            self.rules.len(),
            watched_paths.len()
        );

        // Event processing loop
        loop {
            tokio::select! {
                Some(event) = fs_rx.recv() => {
                    self.process_event(&event, &debouncer, &action_tx).await;
                }
                _ = cancel.changed() => {
                    if *cancel.borrow() {
                        tracing::info!("File system reactor cancelled");
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    async fn process_event(
        &self,
        event: &Event,
        debouncer: &Arc<RwLock<EventDebouncer>>,
        action_tx: &mpsc::Sender<(ReactorRule, String, FsEventType)>,
    ) {
        let event_type = match event.kind {
            EventKind::Create(_) => FsEventType::Created,
            EventKind::Modify(_) => FsEventType::Modified,
            EventKind::Remove(_) => FsEventType::Deleted,
            _ => return,
        };

        for path in &event.paths {
            let path_str = path.to_string_lossy().to_string();

            for rule in &self.rules {
                if !rule.enabled {
                    continue;
                }
                if !rule.matches_event_type(&event_type) {
                    continue;
                }
                if !rule.matches_path(&path_str) {
                    continue;
                }

                // Check debounce
                let key = format!("{}:{}", rule.id, path_str);
                let should_process = debouncer
                    .write()
                    .await
                    .should_process(&key, rule.debounce_ms);

                if !should_process {
                    continue;
                }

                tracing::debug!("Rule '{}' matched: {} {}", rule.name, event_type, path_str);

                if let Err(e) = action_tx
                    .send((rule.clone(), path_str.clone(), event_type.clone()))
                    .await
                {
                    tracing::error!("Failed to dispatch reactor action: {e}");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fs_reactor_new_accepts_config() {
        let config = ReactorConfig::default();
        let reactor = FsReactor::new(config, vec![]);
        assert!(reactor.rules.is_empty());
    }
}
