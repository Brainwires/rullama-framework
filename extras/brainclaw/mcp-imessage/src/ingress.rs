//! Polling ingress loop for the BlueBubbles bridge.
//!
//! The loop polls `/api/v1/message` at a configurable interval, filters
//! the result against a per-chat `last_guid` cursor, and forwards
//! brand-new messages to the gateway via an `mpsc` channel. The cursor
//! is persisted after each successful batch so a restart doesn't
//! re-forward messages.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use rand::Rng;
use tokio::sync::{Mutex, mpsc};

use brainwires_network::channels::ChannelEvent;

use crate::imessage::{BbMessage, ImessageChannel, bb_to_channel};
use crate::state::CursorState;

/// Maximum batch size fetched per poll.
pub const POLL_LIMIT: u32 = 100;

/// Handle used to drive the ingress loop.
pub struct Ingress {
    channel: Arc<ImessageChannel>,
    watched: Vec<String>,
    poll_interval: Duration,
    cursor_path: PathBuf,
    cursor: Arc<Mutex<CursorState>>,
}

impl Ingress {
    /// Construct a new ingress wrapper.
    ///
    /// `watched` is the set of chat guids to forward. Empty means all.
    /// The cursor is loaded lazily from `cursor_path`.
    pub fn new(
        channel: Arc<ImessageChannel>,
        watched: Vec<String>,
        poll_interval: Duration,
        cursor_path: PathBuf,
    ) -> Result<Self> {
        let cursor = CursorState::load(&cursor_path)?;
        Ok(Self {
            channel,
            watched,
            poll_interval,
            cursor_path,
            cursor: Arc::new(Mutex::new(cursor)),
        })
    }

    /// Run the polling loop until either `shutdown` fires or the
    /// `event_tx` is dropped.
    pub async fn run(
        self,
        event_tx: mpsc::Sender<ChannelEvent>,
        shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> Result<()> {
        let mut shutdown = shutdown;
        // Jitter the first poll to avoid stampedes when many adapters
        // start in lockstep.
        let jitter_ms: u64 = { rand::rng().random_range(0..500) };
        tokio::time::sleep(Duration::from_millis(jitter_ms)).await;

        let mut backoff = Duration::from_millis(500);
        loop {
            if *shutdown.borrow() {
                break;
            }
            match self.tick(&event_tx).await {
                Ok(_) => backoff = Duration::from_millis(500),
                Err(e) => {
                    tracing::warn!(error = %e, "imessage ingress tick failed");
                    backoff = std::cmp::min(backoff.saturating_mul(2), Duration::from_secs(30));
                }
            }

            let sleep = if backoff > self.poll_interval {
                backoff
            } else {
                self.poll_interval
            };

            tokio::select! {
                _ = tokio::time::sleep(sleep) => {}
                changed = shutdown.changed() => {
                    if changed.is_ok() && *shutdown.borrow() {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    /// Execute a single polling cycle — fetch, filter, forward, persist.
    pub async fn tick(&self, event_tx: &mpsc::Sender<ChannelEvent>) -> Result<usize> {
        let after = self.earliest_cursor().await;
        let batch = self
            .channel
            .poll_messages(POLL_LIMIT, after.as_deref())
            .await?;
        let mut forwarded = 0usize;
        let mut cursor = self.cursor.lock().await;
        for bb in batch.iter() {
            if !self.should_watch(bb) {
                continue;
            }
            let chat_guid = match bb.chats.first() {
                Some(c) => &c.guid,
                None => continue,
            };
            if cursor
                .last_guid
                .get(chat_guid)
                .is_some_and(|g| g == &bb.guid)
            {
                continue;
            }
            if let Some(msg) = bb_to_channel(bb) {
                if event_tx
                    .send(ChannelEvent::MessageReceived(msg))
                    .await
                    .is_err()
                {
                    break;
                }
                forwarded += 1;
            }
            cursor.last_guid.insert(chat_guid.clone(), bb.guid.clone());
        }
        cursor.save(&self.cursor_path)?;
        Ok(forwarded)
    }

    async fn earliest_cursor(&self) -> Option<String> {
        // BlueBubbles' `after` parameter is global across chats. We pass
        // the lexicographically-largest known guid across all watched
        // chats — if we're only tracking some chats, we might receive a
        // few already-seen messages, which are then filtered out below
        // via the per-chat `last_guid` map.
        let c = self.cursor.lock().await;
        c.last_guid.values().max().cloned()
    }

    fn should_watch(&self, msg: &BbMessage) -> bool {
        if self.watched.is_empty() {
            return true;
        }
        let Some(chat) = msg.chats.first() else {
            return false;
        };
        self.watched.iter().any(|g| g == &chat.guid)
    }

    /// Snapshot of the in-memory cursor state — exposed for tests and
    /// diagnostics.
    pub async fn cursor_snapshot(&self) -> CursorState {
        self.cursor.lock().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_watch_empty_is_all() {
        let ch = Arc::new(ImessageChannel::new("http://x", "pw"));
        let td = tempfile::tempdir().unwrap();
        let ing = Ingress::new(
            ch,
            vec![],
            Duration::from_millis(10),
            td.path().join("c.json"),
        )
        .unwrap();
        let msg = BbMessage {
            guid: "g".into(),
            text: Some("hi".into()),
            date_created_ms: Some(1),
            chats: vec![crate::imessage::BbChat {
                guid: "chat-1".into(),
            }],
            handle: None,
            is_from_me: false,
        };
        assert!(ing.should_watch(&msg));
    }

    #[test]
    fn should_watch_filters_out_non_matching() {
        let ch = Arc::new(ImessageChannel::new("http://x", "pw"));
        let td = tempfile::tempdir().unwrap();
        let ing = Ingress::new(
            ch,
            vec!["chat-1".into()],
            Duration::from_millis(10),
            td.path().join("c.json"),
        )
        .unwrap();
        let msg = BbMessage {
            guid: "g".into(),
            text: Some("x".into()),
            date_created_ms: None,
            chats: vec![crate::imessage::BbChat {
                guid: "chat-2".into(),
            }],
            handle: None,
            is_from_me: false,
        };
        assert!(!ing.should_watch(&msg));
    }
}
