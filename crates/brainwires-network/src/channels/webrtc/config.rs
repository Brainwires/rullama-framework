//! Serde-serializable WebRTC configuration types.
//!
//! All types here are plain Rust values with no dependency on the `webrtc` crate
//! in their public API — the single bridge point is [`WebRtcConfig::to_rtc_configuration`].

use serde::{Deserialize, Serialize};

// ── ICE ───────────────────────────────────────────────────────────────────────

/// A STUN or TURN server definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IceServer {
    /// One or more server URLs, e.g. `["stun:stun.l.google.com:19302"]`.
    pub urls: Vec<String>,
    /// TURN username (leave `None` for STUN-only servers).
    pub username: Option<String>,
    /// TURN credential (leave `None` for STUN-only servers).
    pub credential: Option<String>,
}

/// Controls which ICE candidate types are gathered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum IceTransportPolicy {
    /// Gather all candidate types: host, server-reflexive, relay.
    #[default]
    All,
    /// Only gather relay (TURN) candidates — forces all traffic through TURN.
    Relay,
}

// ── DTLS ──────────────────────────────────────────────────────────────────────

/// DTLS role for the initial certificate handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DtlsRole {
    /// Automatically negotiate DTLS role (recommended).
    #[default]
    Auto,
    /// Act as the DTLS client (initiates the handshake).
    Client,
    /// Act as the DTLS server (waits for the handshake).
    Server,
}

// ── Codecs ────────────────────────────────────────────────────────────────────

/// Preferred video codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoCodec {
    Vp8,
    Vp9,
    H264,
    Av1,
}

/// Preferred audio codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioCodec {
    Opus,
    G711Ulaw,
    G711Alaw,
}

/// Codec preferences passed to the SDP negotiation.
///
/// Listed in priority order; the first entry that the remote supports is chosen.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodecPreferences {
    /// Video codec priority list.
    pub video: Vec<VideoCodec>,
    /// Audio codec priority list.
    pub audio: Vec<AudioCodec>,
}

impl CodecPreferences {
    /// Sensible defaults: VP8+Opus primary, VP9+H264 fallbacks.
    pub fn default_webrtc() -> Self {
        Self {
            video: vec![VideoCodec::Vp8, VideoCodec::Vp9, VideoCodec::H264],
            audio: vec![AudioCodec::Opus],
        }
    }
}

// ── Bandwidth ─────────────────────────────────────────────────────────────────

/// Bandwidth constraints forwarded to the GCC congestion controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthConstraints {
    /// Minimum bitrate in bps.
    pub min_bps: u32,
    /// Initial / target bitrate in bps.
    pub start_bps: u32,
    /// Maximum bitrate in bps.
    pub max_bps: u32,
}

impl Default for BandwidthConstraints {
    fn default() -> Self {
        Self {
            min_bps: 30_000,
            start_bps: 300_000,
            max_bps: 2_500_000,
        }
    }
}

// ── Top-level config ──────────────────────────────────────────────────────────

/// Top-level WebRTC configuration.
///
/// Fully serde-serializable so it can be stored in gateway config files,
/// passed over the A2A protocol, or embedded in agent capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebRtcConfig {
    /// ICE servers (STUN/TURN).
    pub ice_servers: Vec<IceServer>,
    /// Controls which ICE candidate types are gathered.
    pub ice_transport_policy: IceTransportPolicy,
    /// Codec preferences for SDP negotiation.
    pub codec_preferences: CodecPreferences,
    /// Bandwidth constraints for the GCC congestion controller.
    pub bandwidth: BandwidthConstraints,
    /// DTLS role.
    pub dtls_role: DtlsRole,
    /// Obfuscate LAN IP addresses using mDNS `.local` hostnames.
    pub mdns_enabled: bool,
    /// Also gather TCP ICE candidates (useful when UDP is blocked by firewalls).
    pub tcp_candidates_enabled: bool,
    /// Local addresses to bind to for ICE candidate gathering.
    ///
    /// Defaults to `["0.0.0.0:0"]` (all interfaces, OS-assigned port).
    /// Use specific IPs to restrict to a particular network interface.
    pub bind_addresses: Vec<String>,
}

impl Default for WebRtcConfig {
    fn default() -> Self {
        Self {
            ice_servers: vec![IceServer {
                urls: vec!["stun:stun.l.google.com:19302".to_string()],
                username: None,
                credential: None,
            }],
            ice_transport_policy: IceTransportPolicy::All,
            codec_preferences: CodecPreferences::default_webrtc(),
            bandwidth: BandwidthConstraints::default(),
            dtls_role: DtlsRole::Auto,
            mdns_enabled: false,
            tcp_candidates_enabled: true,
            bind_addresses: vec!["0.0.0.0:0".to_string()],
        }
    }
}

impl WebRtcConfig {
    /// Convert to the `webrtc` crate's `RTCConfiguration`.
    ///
    /// This is the **sole** bridge between our serde types and the `webrtc` crate types.
    pub fn to_rtc_configuration(&self) -> webrtc::peer_connection::RTCConfiguration {
        use webrtc::peer_connection::{
            RTCConfigurationBuilder, RTCIceServer, RTCIceTransportPolicy,
        };

        let ice_servers: Vec<RTCIceServer> = self
            .ice_servers
            .iter()
            .map(|s| RTCIceServer {
                urls: s.urls.clone(),
                username: s.username.clone().unwrap_or_default(),
                credential: s.credential.clone().unwrap_or_default(),
            })
            .collect();

        RTCConfigurationBuilder::new()
            .with_ice_servers(ice_servers)
            .with_ice_transport_policy(match self.ice_transport_policy {
                IceTransportPolicy::All => RTCIceTransportPolicy::All,
                IceTransportPolicy::Relay => RTCIceTransportPolicy::Relay,
            })
            .build()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn webrtc_config_serde_roundtrip() {
        let config = WebRtcConfig {
            ice_servers: vec![
                IceServer {
                    urls: vec!["stun:stun.example.com:3478".to_string()],
                    username: None,
                    credential: None,
                },
                IceServer {
                    urls: vec!["turn:turn.example.com:3478".to_string()],
                    username: Some("user".to_string()),
                    credential: Some("pass".to_string()),
                },
            ],
            ice_transport_policy: IceTransportPolicy::Relay,
            codec_preferences: CodecPreferences::default_webrtc(),
            bandwidth: BandwidthConstraints {
                min_bps: 50_000,
                start_bps: 500_000,
                max_bps: 5_000_000,
            },
            dtls_role: DtlsRole::Client,
            mdns_enabled: true,
            tcp_candidates_enabled: false,
            bind_addresses: vec!["192.168.1.1:0".to_string()],
        };

        let json = serde_json::to_string(&config).unwrap();
        let rt: WebRtcConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(rt.ice_servers.len(), 2);
        assert_eq!(rt.ice_transport_policy, IceTransportPolicy::Relay);
        assert_eq!(rt.dtls_role, DtlsRole::Client);
        assert!(rt.mdns_enabled);
        assert!(!rt.tcp_candidates_enabled);
        assert_eq!(rt.bandwidth.min_bps, 50_000);
        assert_eq!(rt.bind_addresses, vec!["192.168.1.1:0"]);
    }

    #[test]
    fn bandwidth_constraints_defaults() {
        let bw = BandwidthConstraints::default();
        assert_eq!(bw.min_bps, 30_000);
        assert_eq!(bw.start_bps, 300_000);
        assert_eq!(bw.max_bps, 2_500_000);
    }

    #[test]
    fn codec_preferences_default_webrtc() {
        let prefs = CodecPreferences::default_webrtc();
        assert_eq!(prefs.video[0], VideoCodec::Vp8);
        assert_eq!(prefs.audio[0], AudioCodec::Opus);
    }
}
