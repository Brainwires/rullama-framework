//! Protocol Telemetry and Metrics
//!
//! Provides observability for the remote control protocol,
//! tracking message latency, throughput, and error rates.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Maximum number of latency samples to keep for histogram
const MAX_LATENCY_SAMPLES: usize = 1000;

/// Protocol metrics for observability
#[derive(Debug, Default)]
pub struct ProtocolMetrics {
    /// Message latency samples (in milliseconds)
    latency_samples: RwLock<VecDeque<u64>>,

    /// Command roundtrip samples (in milliseconds)
    roundtrip_samples: RwLock<VecDeque<u64>>,

    /// Total messages sent
    messages_sent: AtomicU64,

    /// Total messages failed
    messages_failed: AtomicU64,

    /// Total bytes sent
    bytes_sent: AtomicU64,

    /// Total bytes received
    bytes_received: AtomicU64,

    /// Total bytes before compression
    bytes_uncompressed: AtomicU64,

    /// Total bytes after compression
    bytes_compressed: AtomicU64,

    /// Connection start time
    connection_start: RwLock<Option<Instant>>,

    /// Last activity time
    last_activity: RwLock<Option<Instant>>,
}

impl ProtocolMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Record connection start
    pub fn record_connection_start(&self) {
        let mut start = self
            .connection_start
            .write()
            .expect("metrics lock poisoned");
        *start = Some(Instant::now());
        let mut activity = self.last_activity.write().expect("metrics lock poisoned");
        *activity = Some(Instant::now());
    }

    /// Record message sent
    pub fn record_message_sent(&self, bytes: u64) {
        self.messages_sent.fetch_add(1, Ordering::Relaxed);
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
        let mut activity = self.last_activity.write().expect("metrics lock poisoned");
        *activity = Some(Instant::now());
    }

    /// Record message failed
    pub fn record_message_failed(&self) {
        self.messages_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record bytes received
    pub fn record_bytes_received(&self, bytes: u64) {
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
        let mut activity = self.last_activity.write().expect("metrics lock poisoned");
        *activity = Some(Instant::now());
    }

    /// Record compression ratio (uncompressed -> compressed)
    pub fn record_compression(&self, uncompressed: u64, compressed: u64) {
        self.bytes_uncompressed
            .fetch_add(uncompressed, Ordering::Relaxed);
        self.bytes_compressed
            .fetch_add(compressed, Ordering::Relaxed);
    }

    /// Record message latency (one-way)
    pub fn record_latency(&self, latency: Duration) {
        let ms = latency.as_millis() as u64;
        let mut samples = self.latency_samples.write().expect("metrics lock poisoned");
        if samples.len() >= MAX_LATENCY_SAMPLES {
            samples.pop_front();
        }
        samples.push_back(ms);
    }

    /// Record command roundtrip time
    pub fn record_roundtrip(&self, roundtrip: Duration) {
        let ms = roundtrip.as_millis() as u64;
        let mut samples = self
            .roundtrip_samples
            .write()
            .expect("metrics lock poisoned");
        if samples.len() >= MAX_LATENCY_SAMPLES {
            samples.pop_front();
        }
        samples.push_back(ms);
    }

    /// Get current metrics snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        let latency_samples = self.latency_samples.read().expect("metrics lock poisoned");
        let roundtrip_samples = self
            .roundtrip_samples
            .read()
            .expect("metrics lock poisoned");
        let connection_start = self.connection_start.read().expect("metrics lock poisoned");
        let last_activity = self.last_activity.read().expect("metrics lock poisoned");

        let uptime_secs = connection_start.map(|s| s.elapsed().as_secs()).unwrap_or(0);

        let idle_secs = last_activity.map(|s| s.elapsed().as_secs()).unwrap_or(0);

        let bytes_uncompressed = self.bytes_uncompressed.load(Ordering::Relaxed);
        let bytes_compressed = self.bytes_compressed.load(Ordering::Relaxed);
        let compression_ratio = if bytes_uncompressed > 0 {
            bytes_compressed as f64 / bytes_uncompressed as f64
        } else {
            1.0
        };

        MetricsSnapshot {
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            messages_failed: self.messages_failed.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            compression_ratio,
            latency_p50: percentile(&latency_samples, 50),
            latency_p95: percentile(&latency_samples, 95),
            latency_p99: percentile(&latency_samples, 99),
            roundtrip_p50: percentile(&roundtrip_samples, 50),
            roundtrip_p95: percentile(&roundtrip_samples, 95),
            roundtrip_p99: percentile(&roundtrip_samples, 99),
            uptime_secs,
            idle_secs,
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.latency_samples
            .write()
            .expect("metrics lock poisoned")
            .clear();
        self.roundtrip_samples
            .write()
            .expect("metrics lock poisoned")
            .clear();
        self.messages_sent.store(0, Ordering::Relaxed);
        self.messages_failed.store(0, Ordering::Relaxed);
        self.bytes_sent.store(0, Ordering::Relaxed);
        self.bytes_received.store(0, Ordering::Relaxed);
        self.bytes_uncompressed.store(0, Ordering::Relaxed);
        self.bytes_compressed.store(0, Ordering::Relaxed);
        *self
            .connection_start
            .write()
            .expect("metrics lock poisoned") = None;
        *self.last_activity.write().expect("metrics lock poisoned") = None;
    }
}

/// Calculate percentile from samples
fn percentile(samples: &VecDeque<u64>, p: u32) -> Option<u64> {
    if samples.is_empty() {
        return None;
    }

    let mut sorted: Vec<_> = samples.iter().copied().collect();
    sorted.sort_unstable();

    let index = ((p as f64 / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    Some(sorted[index])
}

/// Snapshot of protocol metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages that failed
    pub messages_failed: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Compression ratio (1.0 = no compression, <1.0 = good compression)
    pub compression_ratio: f64,
    /// Latency 50th percentile (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_p50: Option<u64>,
    /// Latency 95th percentile (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_p95: Option<u64>,
    /// Latency 99th percentile (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_p99: Option<u64>,
    /// Command roundtrip 50th percentile (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roundtrip_p50: Option<u64>,
    /// Command roundtrip 95th percentile (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roundtrip_p95: Option<u64>,
    /// Command roundtrip 99th percentile (ms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roundtrip_p99: Option<u64>,
    /// Connection uptime in seconds
    pub uptime_secs: u64,
    /// Time since last activity in seconds
    pub idle_secs: u64,
}

impl Default for MetricsSnapshot {
    fn default() -> Self {
        Self {
            messages_sent: 0,
            messages_failed: 0,
            bytes_sent: 0,
            bytes_received: 0,
            compression_ratio: 1.0,
            latency_p50: None,
            latency_p95: None,
            latency_p99: None,
            roundtrip_p50: None,
            roundtrip_p95: None,
            roundtrip_p99: None,
            uptime_secs: 0,
            idle_secs: 0,
        }
    }
}

/// Connection quality assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionQuality {
    /// Excellent (< 50ms latency, < 1% error rate)
    Excellent,
    /// Good (< 100ms latency, < 5% error rate)
    Good,
    /// Fair (< 250ms latency, < 10% error rate)
    Fair,
    /// Poor (> 250ms latency or > 10% error rate)
    Poor,
    /// Unknown (not enough data)
    Unknown,
}

impl MetricsSnapshot {
    /// Assess connection quality based on metrics
    pub fn connection_quality(&self) -> ConnectionQuality {
        // Need enough samples
        if self.messages_sent < 10 {
            return ConnectionQuality::Unknown;
        }

        // Calculate error rate
        let error_rate = if self.messages_sent > 0 {
            self.messages_failed as f64 / self.messages_sent as f64
        } else {
            0.0
        };

        // Use p95 latency for quality assessment
        let latency = self.latency_p95.unwrap_or(0);

        if error_rate > 0.10 || latency > 250 {
            ConnectionQuality::Poor
        } else if error_rate > 0.05 || latency > 100 {
            ConnectionQuality::Fair
        } else if error_rate > 0.01 || latency > 50 {
            ConnectionQuality::Good
        } else {
            ConnectionQuality::Excellent
        }
    }

    /// Calculate throughput in bytes per second
    pub fn throughput_bps(&self) -> f64 {
        if self.uptime_secs > 0 {
            (self.bytes_sent + self.bytes_received) as f64 / self.uptime_secs as f64
        } else {
            0.0
        }
    }

    /// Calculate message rate per second
    pub fn messages_per_second(&self) -> f64 {
        if self.uptime_secs > 0 {
            self.messages_sent as f64 / self.uptime_secs as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = ProtocolMetrics::new();
        metrics.record_connection_start();

        metrics.record_message_sent(100);
        metrics.record_message_sent(200);
        metrics.record_message_failed();
        metrics.record_bytes_received(150);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.messages_sent, 2);
        assert_eq!(snapshot.messages_failed, 1);
        assert_eq!(snapshot.bytes_sent, 300);
        assert_eq!(snapshot.bytes_received, 150);
    }

    #[test]
    fn test_latency_percentiles() {
        let metrics = ProtocolMetrics::new();

        // Add 100 samples from 1-100ms
        for i in 1..=100 {
            metrics.record_latency(Duration::from_millis(i));
        }

        let snapshot = metrics.snapshot();
        // Percentile calculation may vary slightly due to rounding
        let p50 = snapshot.latency_p50.unwrap();
        let p95 = snapshot.latency_p95.unwrap();
        let p99 = snapshot.latency_p99.unwrap();
        assert!(
            (49..=51).contains(&p50),
            "p50 should be around 50, got {}",
            p50
        );
        assert!(
            (94..=96).contains(&p95),
            "p95 should be around 95, got {}",
            p95
        );
        assert!(
            (98..=100).contains(&p99),
            "p99 should be around 99, got {}",
            p99
        );
    }

    #[test]
    fn test_compression_ratio() {
        let metrics = ProtocolMetrics::new();
        metrics.record_compression(1000, 400); // 60% compression

        let snapshot = metrics.snapshot();
        assert!((snapshot.compression_ratio - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_connection_quality() {
        let mut snapshot = MetricsSnapshot::default();

        // Not enough data
        assert_eq!(snapshot.connection_quality(), ConnectionQuality::Unknown);

        // Good connection
        snapshot.messages_sent = 100;
        snapshot.messages_failed = 0;
        snapshot.latency_p95 = Some(30);
        assert_eq!(snapshot.connection_quality(), ConnectionQuality::Excellent);

        // Fair connection
        snapshot.latency_p95 = Some(120);
        assert_eq!(snapshot.connection_quality(), ConnectionQuality::Fair);

        // Poor connection
        snapshot.messages_failed = 15; // 15% error rate
        assert_eq!(snapshot.connection_quality(), ConnectionQuality::Poor);
    }
}
