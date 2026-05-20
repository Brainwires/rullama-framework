//! Per-room polling ingress loop.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use rand::Rng;
use tokio::sync::{Mutex, mpsc};

use brainwires_network::channels::ChannelEvent;

use crate::nextcloud_talk::{NextcloudTalkChannel, spreed_to_channel};
use crate::state::CursorState;

/// Handle driving the polling loop.
pub struct Ingress {
    channel: Arc<NextcloudTalkChannel>,
    rooms: Vec<String>,
    poll_interval: Duration,
    cursor_path: PathBuf,
    cursor: Arc<Mutex<CursorState>>,
}

impl Ingress {
    /// Construct a new ingress. Loads (or creates) a cursor from
    /// `cursor_path`.
    pub fn new(
        channel: Arc<NextcloudTalkChannel>,
        rooms: Vec<String>,
        poll_interval: Duration,
        cursor_path: PathBuf,
    ) -> Result<Self> {
        let cursor = CursorState::load(&cursor_path)?;
        Ok(Self {
            channel,
            rooms,
            poll_interval,
            cursor_path,
            cursor: Arc::new(Mutex::new(cursor)),
        })
    }

    /// Run the polling loop until `shutdown` fires or the sender drops.
    pub async fn run(
        self,
        event_tx: mpsc::Sender<ChannelEvent>,
        mut shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> Result<()> {
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
                    tracing::warn!(error = %e, "nextcloud-talk ingress tick failed");
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

    /// Single polling cycle across all watched rooms.
    pub async fn tick(&self, event_tx: &mpsc::Sender<ChannelEvent>) -> Result<usize> {
        let mut forwarded = 0usize;
        let host = self.channel.host_fragment().to_string();
        for room in &self.rooms {
            let last = {
                let c = self.cursor.lock().await;
                c.last_message_id.get(room).copied().unwrap_or(0)
            };
            let msgs = match self.channel.poll_room(room, last).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(room = %room, error = %e, "poll_room failed");
                    continue;
                }
            };
            let mut new_cursor = last;
            for m in msgs {
                if m.id <= last {
                    continue;
                }
                if let Some(cm) = spreed_to_channel(&m, room, &host) {
                    if event_tx
                        .send(ChannelEvent::MessageReceived(cm))
                        .await
                        .is_err()
                    {
                        break;
                    }
                    forwarded += 1;
                }
                new_cursor = new_cursor.max(m.id);
            }
            if new_cursor > last {
                let mut c = self.cursor.lock().await;
                c.last_message_id.insert(room.clone(), new_cursor);
            }
        }
        self.cursor.lock().await.save(&self.cursor_path)?;
        Ok(forwarded)
    }

    /// Snapshot of the in-memory cursor state — exposed for tests and
    /// diagnostics.
    pub async fn cursor_snapshot(&self) -> CursorState {
        self.cursor.lock().await.clone()
    }
}
