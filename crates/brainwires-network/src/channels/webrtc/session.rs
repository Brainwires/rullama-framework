//! WebRTC session management — one `WebRtcSession` per `PeerConnection`.
//!
//! # Lifecycle
//!
//! ```text
//! WebRtcSession::new(config, conversation)
//!   → session.open().await?
//!   → session.add_audio_track(AudioCodec::Opus).await?
//!   → sdp = session.create_offer().await?
//!   // send sdp via WebRtcSignaling::send_signaling
//!   → session.set_remote_description(answer_sdp, SdpType::Answer).await?
//!   // ICE candidates flow through on_ice_candidate → ChannelEvent::IceCandidate broadcasts
//!   → session.add_ice_candidate(candidate, sdp_mid, sdp_mline_index).await?
//!   // PeerConnectionState::Connected reached
//!   → session.close().await?
//! ```
//!
//! Subscribe to all events with [`WebRtcSession::subscribe`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use rtc::rtp_transceiver::SSRC;
use rtc::rtp_transceiver::rtp_sender::{
    RTCRtpCodec, RTCRtpCodingParameters, RTCRtpEncodingParameters, RtpCodecKind,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast};
use uuid::Uuid;
use webrtc::data_channel::DataChannel as WrtcDataChannel;
use webrtc::data_channel::DataChannelEvent;
use webrtc::media_stream::MediaStreamTrack;
use webrtc::media_stream::track_local::TrackLocal;
use webrtc::media_stream::track_local::static_sample::TrackLocalStaticSample;
use webrtc::media_stream::track_remote::TrackRemote;
use webrtc::media_stream::track_remote::TrackRemoteEvent;
use webrtc::peer_connection::{
    MediaEngine, MulticastDnsMode, PeerConnection, PeerConnectionBuilder,
    PeerConnectionEventHandler, RTCIceConnectionState, RTCIceGatheringState,
    RTCPeerConnectionIceEvent, RTCPeerConnectionState, RTCSessionDescription, RTCSignalingState,
    Registry, SettingEngine, register_default_interceptors,
};

use super::super::events::ChannelEvent;
use super::super::identity::ConversationId;

use super::config::DtlsRole;
use super::config::{AudioCodec, VideoCodec, WebRtcConfig};
use super::track::{
    AudioTrack, DataChannel, DataChannelConfig, DataChannelMessage, MediaTrack, RemoteTrack,
    TrackDirection, TrackId, VideoTrack,
};

// ── Identifier ────────────────────────────────────────────────────────────────

/// A unique identifier for a WebRTC session (one per `PeerConnection`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct WebRtcSessionId(pub Uuid);

impl WebRtcSessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for WebRtcSessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for WebRtcSessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── State enums ───────────────────────────────────────────────────────────────

/// Offer/answer negotiation state (mirrors the W3C `RTCSignalingState`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum SignalingState {
    Stable,
    HaveLocalOffer,
    HaveRemoteOffer,
    HaveLocalPrAnswer,
    HaveRemotePrAnswer,
    Closed,
}

/// Overall PeerConnection state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum PeerConnectionState {
    New,
    Connecting,
    Connected,
    Disconnected,
    Failed,
    Closed,
}

/// ICE connection state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum IceConnectionState {
    New,
    Checking,
    Connected,
    Completed,
    Failed,
    Disconnected,
    Closed,
}

/// Whether an SDP description is an offer or an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpType {
    Offer,
    Answer,
}

// ── Internal mutable state ────────────────────────────────────────────────────

struct SessionState {
    signaling_state: SignalingState,
    peer_connection_state: PeerConnectionState,
    ice_connection_state: IceConnectionState,
    local_tracks: Vec<MediaTrack>,
    remote_tracks: HashMap<TrackId, Arc<RemoteTrack>>,
}

// ── Event handler ─────────────────────────────────────────────────────────────

/// Implements `PeerConnectionEventHandler` for a `WebRtcSession`.
///
/// All fields must be `Arc`-wrapped so they can be captured by the handler,
/// which is moved into the `PeerConnectionBuilder`.
struct SessionEventHandler {
    session_id: WebRtcSessionId,
    conversation: ConversationId,
    event_tx: broadcast::Sender<ChannelEvent>,
    state: Arc<RwLock<SessionState>>,
}

#[async_trait::async_trait]
impl PeerConnectionEventHandler for SessionEventHandler {
    async fn on_ice_candidate(&self, event: RTCPeerConnectionIceEvent) {
        if let Ok(init) = event.candidate.to_json() {
            let _ = self.event_tx.send(ChannelEvent::IceCandidate {
                session_id: self.session_id.clone(),
                candidate: init.candidate,
                sdp_mid: init.sdp_mid,
                sdp_mline_index: init.sdp_mline_index,
                conversation: self.conversation.clone(),
            });
        }
    }

    async fn on_connection_state_change(&self, state: RTCPeerConnectionState) {
        let mapped = match state {
            RTCPeerConnectionState::New => PeerConnectionState::New,
            RTCPeerConnectionState::Connecting => PeerConnectionState::Connecting,
            RTCPeerConnectionState::Connected => PeerConnectionState::Connected,
            RTCPeerConnectionState::Disconnected => PeerConnectionState::Disconnected,
            RTCPeerConnectionState::Failed => PeerConnectionState::Failed,
            RTCPeerConnectionState::Closed => PeerConnectionState::Closed,
            _ => return,
        };
        self.state.write().await.peer_connection_state = mapped.clone();
        let _ = self
            .event_tx
            .send(ChannelEvent::PeerConnectionStateChanged {
                session_id: self.session_id.clone(),
                state: mapped,
                conversation: self.conversation.clone(),
            });
    }

    async fn on_ice_connection_state_change(&self, state: RTCIceConnectionState) {
        let mapped = match state {
            RTCIceConnectionState::New => IceConnectionState::New,
            RTCIceConnectionState::Checking => IceConnectionState::Checking,
            RTCIceConnectionState::Connected => IceConnectionState::Connected,
            RTCIceConnectionState::Completed => IceConnectionState::Completed,
            RTCIceConnectionState::Failed => IceConnectionState::Failed,
            RTCIceConnectionState::Disconnected => IceConnectionState::Disconnected,
            RTCIceConnectionState::Closed => IceConnectionState::Closed,
            _ => return,
        };
        self.state.write().await.ice_connection_state = mapped.clone();
        let _ = self.event_tx.send(ChannelEvent::IceConnectionStateChanged {
            session_id: self.session_id.clone(),
            state: mapped,
            conversation: self.conversation.clone(),
        });
    }

    async fn on_ice_gathering_state_change(&self, state: RTCIceGatheringState) {
        if state == RTCIceGatheringState::Complete {
            let _ = self.event_tx.send(ChannelEvent::IceGatheringComplete {
                session_id: self.session_id.clone(),
                conversation: self.conversation.clone(),
            });
        }
    }

    async fn on_signaling_state_change(&self, state: RTCSignalingState) {
        let mapped = match state {
            RTCSignalingState::Stable => SignalingState::Stable,
            RTCSignalingState::HaveLocalOffer => SignalingState::HaveLocalOffer,
            RTCSignalingState::HaveRemoteOffer => SignalingState::HaveRemoteOffer,
            RTCSignalingState::HaveLocalPranswer => SignalingState::HaveLocalPrAnswer,
            RTCSignalingState::HaveRemotePranswer => SignalingState::HaveRemotePrAnswer,
            RTCSignalingState::Closed => SignalingState::Closed,
            _ => return,
        };
        self.state.write().await.signaling_state = mapped.clone();
        let _ = self.event_tx.send(ChannelEvent::SignalingStateChanged {
            session_id: self.session_id.clone(),
            state: mapped,
            conversation: self.conversation.clone(),
        });
    }

    async fn on_track(&self, track: Arc<dyn TrackRemote>) {
        let kind = match track.kind().await {
            RtpCodecKind::Audio => "audio",
            RtpCodecKind::Video => "video",
            _ => "unknown",
        };
        let track_id = TrackId::new(track.track_id().await.to_string());
        let ssrcs = track.ssrcs().await;
        let codec = if let Some(&ssrc) = ssrcs.first() {
            track.codec(ssrc).await.map(|c| c.mime_type.clone())
        } else {
            None
        };

        // Store the remote track so callers can read RTP frames via get_remote_track().
        let remote_track = Arc::new(RemoteTrack::new(
            track_id.clone(),
            kind.to_string(),
            codec.clone(),
            track.clone(),
        ));
        self.state
            .write()
            .await
            .remote_tracks
            .insert(track_id.clone(), Arc::clone(&remote_track));

        let _ = self.event_tx.send(ChannelEvent::TrackAdded {
            session_id: self.session_id.clone(),
            track_id: track_id.clone(),
            kind: kind.to_string(),
            codec,
            direction: TrackDirection::RecvOnly,
            conversation: self.conversation.clone(),
        });

        // Spawn a task that watches for OnEnded and emits TrackRemoved.
        let session_id = self.session_id.clone();
        let conversation = self.conversation.clone();
        let event_tx = self.event_tx.clone();
        let state = Arc::clone(&self.state);
        let tid = track_id;
        tokio::spawn(async move {
            loop {
                match track.poll().await {
                    Some(TrackRemoteEvent::OnEnded) | None => break,
                    _ => {}
                }
            }
            state.write().await.remote_tracks.remove(&tid);
            let _ = event_tx.send(ChannelEvent::TrackRemoved {
                session_id,
                track_id: tid,
                conversation,
            });
        });
    }

    async fn on_data_channel(&self, data_channel: Arc<dyn WrtcDataChannel>) {
        // Spawn a poll task that emits `WebRtcDataChannel` events for each inbound message.
        let session_id = self.session_id.clone();
        let conversation = self.conversation.clone();
        let event_tx = self.event_tx.clone();
        let dc = data_channel.clone();
        let label = dc.label().await.unwrap_or_default();
        tokio::spawn(async move {
            loop {
                match dc.poll().await {
                    Some(DataChannelEvent::OnMessage(msg)) => {
                        let dm = if msg.is_string {
                            DataChannelMessage::Text(
                                String::from_utf8_lossy(&msg.data).into_owned(),
                            )
                        } else {
                            DataChannelMessage::Binary(msg.data.to_vec())
                        };
                        let _ = event_tx.send(ChannelEvent::WebRtcDataChannel {
                            session_id: session_id.clone(),
                            channel_label: label.clone(),
                            message: dm,
                            conversation: conversation.clone(),
                        });
                    }
                    Some(DataChannelEvent::OnClose) | None => break,
                    _ => {}
                }
            }
        });
    }
}

// ── WebRtcSession ─────────────────────────────────────────────────────────────

/// Manages a single WebRTC `PeerConnection` with full offer/answer state machine.
///
/// All methods take `&self` so the session can be wrapped in `Arc<WebRtcSession>`
/// and shared across tasks (e.g. a track-writing task and an event-processing task).
pub struct WebRtcSession {
    /// Stable identifier for this session.
    pub id: WebRtcSessionId,
    /// The conversation this session belongs to.
    pub conversation: ConversationId,

    config: WebRtcConfig,
    /// The underlying `PeerConnection`.  `None` until [`open`](Self::open) is called.
    peer_connection: Arc<RwLock<Option<Arc<dyn PeerConnection>>>>,
    /// Shared mutable state, guarded by an async RwLock.
    state: Arc<RwLock<SessionState>>,
    /// Broadcast channel through which all `ChannelEvent`s emitted by this session flow.
    event_tx: broadcast::Sender<ChannelEvent>,
    /// GCC bandwidth estimation handle (available only with the `webrtc-advanced` feature).
    #[cfg(feature = "channels-webrtc-advanced")]
    gcc_handle: std::sync::Mutex<Option<rtc_interceptor::GccHandle>>,
}

impl WebRtcSession {
    // ── Constructor ───────────────────────────────────────────────────────────

    /// Create a new session.  Call [`open`](Self::open) before anything else.
    pub fn new(config: WebRtcConfig, conversation: ConversationId) -> Self {
        let (event_tx, _) = broadcast::channel(128);
        Self {
            id: WebRtcSessionId::new(),
            conversation,
            config,
            peer_connection: Arc::new(RwLock::new(None)),
            state: Arc::new(RwLock::new(SessionState {
                signaling_state: SignalingState::Stable,
                peer_connection_state: PeerConnectionState::New,
                ice_connection_state: IceConnectionState::New,
                local_tracks: Vec::new(),
                remote_tracks: HashMap::new(),
            })),
            event_tx,
            #[cfg(feature = "channels-webrtc-advanced")]
            gcc_handle: std::sync::Mutex::new(None),
        }
    }

    // ── Open ──────────────────────────────────────────────────────────────────

    /// Create the underlying `PeerConnection` and wire all event callbacks.
    ///
    /// Must be called before [`create_offer`](Self::create_offer),
    /// [`set_remote_description`](Self::set_remote_description), or adding tracks.
    pub async fn open(&self) -> Result<()> {
        let rtc_config = self.config.to_rtc_configuration();

        let mut media_engine = MediaEngine::default();
        media_engine
            .register_default_codecs()
            .map_err(|e| anyhow!("register codecs: {e}"))?;

        // Build the base interceptor chain: NACK, RTCP reports, TwccReceiver, simulcast.
        let base_registry = register_default_interceptors(Registry::new(), &mut media_engine)
            .map_err(|e| anyhow!("register interceptors: {e}"))?;

        // Warn when congestion control is unavailable (missing `channels-webrtc-advanced` feature).
        #[cfg(not(feature = "channels-webrtc-advanced"))]
        tracing::warn!(
            session_id = %self.id,
            "WebRtcSession opened without the `channels-webrtc-advanced` feature: \
             GCC congestion control, JitterBuffer, and TwccSender are disabled. \
             Add `features = [\"channels-webrtc-advanced\"]` to enable adaptive bitrate."
        );

        // With `channels-webrtc-advanced`, layer on JitterBuffer + TwccSender + GCC.
        #[cfg(feature = "channels-webrtc-advanced")]
        let registry = {
            use rtc_interceptor::{GccInterceptorBuilder, JitterBufferBuilder, TwccSenderBuilder};
            let bw = &self.config.bandwidth;
            let (gcc_builder_raw, gcc_handle) = GccInterceptorBuilder::new();
            let gcc_builder = gcc_builder_raw
                .with_min_bitrate(bw.min_bps)
                .with_max_bitrate(bw.max_bps);
            if let Ok(mut lock) = self.gcc_handle.lock() {
                *lock = Some(gcc_handle);
            }
            base_registry
                .with(TwccSenderBuilder::new().build())
                .with(gcc_builder.build())
                .with(JitterBufferBuilder::new().build())
        };
        #[cfg(not(feature = "channels-webrtc-advanced"))]
        let registry = base_registry;

        // Apply DTLS role via SettingEngine (only when not Auto).
        let mut setting_engine = SettingEngine::default();
        if self.config.dtls_role != DtlsRole::Auto {
            use rtc::peer_connection::transport::RTCDtlsRole;
            let role = match self.config.dtls_role {
                DtlsRole::Client => RTCDtlsRole::Client,
                DtlsRole::Server => RTCDtlsRole::Server,
                DtlsRole::Auto => unreachable!(),
            };
            setting_engine
                .set_answering_dtls_role(role)
                .map_err(|e| anyhow!("set dtls_role: {e}"))?;
        }
        // mDNS candidate gathering moved off `PeerConnectionBuilder` and onto
        // `SettingEngine` in newer webrtc-rs.
        setting_engine.set_multicast_dns_mode(if self.config.mdns_enabled {
            MulticastDnsMode::QueryAndGather
        } else {
            MulticastDnsMode::Disabled
        });

        let handler = Arc::new(SessionEventHandler {
            session_id: self.id.clone(),
            conversation: self.conversation.clone(),
            event_tx: self.event_tx.clone(),
            state: self.state.clone(),
        });

        let bind_addrs = self.config.bind_addresses.clone();

        let mut builder = PeerConnectionBuilder::new()
            .with_configuration(rtc_config)
            .with_media_engine(media_engine)
            .with_interceptor_registry(registry)
            .with_setting_engine(setting_engine)
            .with_handler(handler as Arc<dyn PeerConnectionEventHandler>)
            .with_udp_addrs(bind_addrs.clone());

        if self.config.tcp_candidates_enabled {
            builder = builder.with_tcp_addrs(bind_addrs);
        }

        let pc: Arc<dyn PeerConnection> = Arc::new(
            builder
                .build()
                .await
                .map_err(|e| anyhow!("build PeerConnection: {e}"))?,
        );

        *self.peer_connection.write().await = Some(pc);
        Ok(())
    }

    // ── Offer / answer ────────────────────────────────────────────────────────

    /// Create an SDP offer and set it as the local description.
    ///
    /// Returns the SDP body to be forwarded to the remote peer via
    /// [`WebRtcSignaling::send_signaling`](super::signaling::WebRtcSignaling::send_signaling).
    pub async fn create_offer(&self) -> Result<String> {
        let pc = self.get_pc().await?;
        let offer = pc
            .create_offer(None)
            .await
            .map_err(|e| anyhow!("create_offer: {e}"))?;
        let sdp = offer.sdp.clone();
        pc.set_local_description(offer)
            .await
            .map_err(|e| anyhow!("set_local_description (offer): {e}"))?;
        self.state.write().await.signaling_state = SignalingState::HaveLocalOffer;
        Ok(sdp)
    }

    /// Create an SDP answer (call after `set_remote_description` with an offer).
    ///
    /// Returns the SDP body to be forwarded to the initiating peer.
    pub async fn create_answer(&self) -> Result<String> {
        let pc = self.get_pc().await?;
        let answer = pc
            .create_answer(None)
            .await
            .map_err(|e| anyhow!("create_answer: {e}"))?;
        let sdp = answer.sdp.clone();
        pc.set_local_description(answer)
            .await
            .map_err(|e| anyhow!("set_local_description (answer): {e}"))?;
        self.state.write().await.signaling_state = SignalingState::Stable;
        Ok(sdp)
    }

    /// Apply a remote SDP description received via signaling.
    pub async fn set_remote_description(&self, sdp: String, sdp_type: SdpType) -> Result<()> {
        let pc = self.get_pc().await?;
        let desc = match sdp_type {
            SdpType::Offer => RTCSessionDescription::offer(sdp).map_err(|e| anyhow!("{e}"))?,
            SdpType::Answer => RTCSessionDescription::answer(sdp).map_err(|e| anyhow!("{e}"))?,
        };
        pc.set_remote_description(desc)
            .await
            .map_err(|e| anyhow!("set_remote_description: {e}"))?;
        let next_state = match sdp_type {
            SdpType::Offer => SignalingState::HaveRemoteOffer,
            SdpType::Answer => SignalingState::Stable,
        };
        self.state.write().await.signaling_state = next_state;
        Ok(())
    }

    // ── ICE ───────────────────────────────────────────────────────────────────

    /// Trickle an ICE candidate received from the remote peer.
    pub async fn add_ice_candidate(
        &self,
        candidate: String,
        sdp_mid: Option<String>,
        sdp_mline_index: Option<u16>,
    ) -> Result<()> {
        use webrtc::peer_connection::RTCIceCandidateInit;
        let pc = self.get_pc().await?;
        pc.add_ice_candidate(RTCIceCandidateInit {
            candidate,
            sdp_mid,
            sdp_mline_index,
            username_fragment: None,
            url: None,
        })
        .await
        .map_err(|e| anyhow!("add_ice_candidate: {e}"))
    }

    // ── ICE restart ───────────────────────────────────────────────────────────

    /// Trigger an ICE restart.
    ///
    /// After this call, create and send a new offer with
    /// [`create_offer`](Self::create_offer).
    pub async fn restart_ice(&self) -> Result<()> {
        let pc = self.get_pc().await?;
        pc.restart_ice()
            .await
            .map_err(|e| anyhow!("restart_ice: {e}"))
    }

    // ── Tracks ────────────────────────────────────────────────────────────────

    /// Add a local audio track to the PeerConnection.
    ///
    /// Must be called before [`create_offer`](Self::create_offer).
    /// Use the returned [`AudioTrack`] to push encoded audio samples.
    pub async fn add_audio_track(&self, codec: AudioCodec) -> Result<AudioTrack> {
        let pc = self.get_pc().await?;

        let (mime_type, clock_rate, channels): (&str, u32, u16) = match codec {
            AudioCodec::Opus => ("audio/opus", 48_000, 2),
            AudioCodec::G711Ulaw => ("audio/PCMU", 8_000, 1),
            AudioCodec::G711Alaw => ("audio/PCMA", 8_000, 1),
        };

        let id = TrackId::new_random();
        let ssrc: SSRC = rand::random::<u32>();

        let rtc_codec = RTCRtpCodec {
            mime_type: mime_type.to_string(),
            clock_rate,
            channels,
            ..Default::default()
        };

        let media_track = MediaStreamTrack::new(
            format!("stream-{id}"),
            format!("track-{id}"),
            format!("audio-{id}"),
            RtpCodecKind::Audio,
            vec![RTCRtpEncodingParameters {
                rtp_coding_parameters: RTCRtpCodingParameters {
                    ssrc: Some(ssrc),
                    ..Default::default()
                },
                codec: rtc_codec,
                ..Default::default()
            }],
        );

        let inner = Arc::new(
            TrackLocalStaticSample::new(media_track)
                .map_err(|e| anyhow!("create audio TrackLocalStaticSample: {e}"))?,
        );
        pc.add_track(Arc::clone(&inner) as Arc<dyn TrackLocal>)
            .await
            .map_err(|e| anyhow!("add_track (audio): {e}"))?;

        let audio = AudioTrack {
            id: id.clone(),
            direction: TrackDirection::SendOnly,
            ssrc,
            inner: inner.clone(),
        };
        self.state
            .write()
            .await
            .local_tracks
            .push(MediaTrack::Audio(AudioTrack {
                id,
                direction: TrackDirection::SendOnly,
                ssrc,
                inner,
            }));
        Ok(audio)
    }

    /// Add a local video track to the PeerConnection.
    ///
    /// Must be called before [`create_offer`](Self::create_offer).
    /// Use the returned [`VideoTrack`] to push encoded video frames.
    pub async fn add_video_track(&self, codec: VideoCodec) -> Result<VideoTrack> {
        let pc = self.get_pc().await?;

        let mime_type: &str = match codec {
            VideoCodec::Vp8 => "video/VP8",
            VideoCodec::Vp9 => "video/VP9",
            VideoCodec::H264 => "video/H264",
            VideoCodec::Av1 => "video/AV1",
        };

        let id = TrackId::new_random();
        let ssrc: SSRC = rand::random::<u32>();

        let rtc_codec = RTCRtpCodec {
            mime_type: mime_type.to_string(),
            clock_rate: 90_000,
            ..Default::default()
        };

        let media_track = MediaStreamTrack::new(
            format!("stream-{id}"),
            format!("track-{id}"),
            format!("video-{id}"),
            RtpCodecKind::Video,
            vec![RTCRtpEncodingParameters {
                rtp_coding_parameters: RTCRtpCodingParameters {
                    ssrc: Some(ssrc),
                    ..Default::default()
                },
                codec: rtc_codec,
                ..Default::default()
            }],
        );

        let inner = Arc::new(
            TrackLocalStaticSample::new(media_track)
                .map_err(|e| anyhow!("create video TrackLocalStaticSample: {e}"))?,
        );
        pc.add_track(Arc::clone(&inner) as Arc<dyn TrackLocal>)
            .await
            .map_err(|e| anyhow!("add_track (video): {e}"))?;

        let video = VideoTrack {
            id: id.clone(),
            direction: TrackDirection::SendOnly,
            ssrc,
            inner: inner.clone(),
        };
        self.state
            .write()
            .await
            .local_tracks
            .push(MediaTrack::Video(VideoTrack {
                id,
                direction: TrackDirection::SendOnly,
                ssrc,
                inner,
            }));
        Ok(video)
    }

    // ── DataChannels ──────────────────────────────────────────────────────────

    /// Open a new WebRTC DataChannel.
    ///
    /// Can be called before or after offer creation.
    pub async fn create_data_channel(&self, config: DataChannelConfig) -> Result<DataChannel> {
        use webrtc::data_channel::RTCDataChannelInit;

        let pc = self.get_pc().await?;
        let init = RTCDataChannelInit {
            ordered: config.ordered,
            max_retransmits: config.max_retransmits,
            protocol: config.protocol.clone().unwrap_or_default(),
            ..Default::default()
        };
        let rtc_dc = pc
            .create_data_channel(&config.label, Some(init))
            .await
            .map_err(|e| anyhow!("create_data_channel '{}': {e}", config.label))?;

        DataChannel::new(rtc_dc).await
    }

    // ── Events ────────────────────────────────────────────────────────────────

    /// Subscribe to [`ChannelEvent`]s emitted by this session.
    ///
    /// Includes `IceCandidate`, `SdpOffer`, `SdpAnswer`, `TrackAdded`,
    /// `PeerConnectionStateChanged`, `IceConnectionStateChanged`, and `WebRtcDataChannel`.
    pub fn subscribe(&self) -> broadcast::Receiver<ChannelEvent> {
        self.event_tx.subscribe()
    }

    // ── State queries ─────────────────────────────────────────────────────────

    pub async fn signaling_state(&self) -> SignalingState {
        self.state.read().await.signaling_state.clone()
    }

    pub async fn peer_connection_state(&self) -> PeerConnectionState {
        self.state.read().await.peer_connection_state.clone()
    }

    pub async fn ice_connection_state(&self) -> IceConnectionState {
        self.state.read().await.ice_connection_state.clone()
    }

    // ── Remote tracks ─────────────────────────────────────────────────────────

    /// Look up a remote track received from the peer by its [`TrackId`].
    ///
    /// Returns `None` if the track has not yet arrived or has already ended.
    /// The [`TrackId`] is available in the [`ChannelEvent::TrackAdded`] payload.
    pub async fn get_remote_track(&self, id: &TrackId) -> Option<Arc<RemoteTrack>> {
        self.state.read().await.remote_tracks.get(id).cloned()
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    /// Return a snapshot of all RTCP statistics for this PeerConnection.
    ///
    /// The returned [`RTCStatsReport`] gives access to per-stream metrics via:
    /// - [`inbound_rtp_streams()`](rtc::statistics::report::RTCStatsReport::inbound_rtp_streams)
    ///   — jitter, packets lost/received, NACK/PLI/FIR counts, jitter buffer stats
    /// - [`outbound_rtp_streams()`](rtc::statistics::report::RTCStatsReport::outbound_rtp_streams)
    ///   — bytes/packets sent, retransmissions, target/encoded bitrate
    /// - [`candidate_pairs()`](rtc::statistics::report::RTCStatsReport::candidate_pairs)
    ///   — round-trip time (RTT), available bandwidth estimates
    ///
    /// Returns `Err` if the session has not been opened yet.
    pub async fn get_stats(&self) -> Result<rtc::statistics::report::RTCStatsReport> {
        let pc = self.get_pc().await?;
        Ok(pc
            .get_stats(Instant::now(), rtc::statistics::StatsSelector::None)
            .await)
    }

    // ── Bandwidth estimation ───────────────────────────────────────────────────

    /// Query the current GCC target bitrate in bps.
    ///
    /// Returns `None` before the first TWCC feedback cycle completes.
    /// Only available with the `webrtc-advanced` feature.
    #[cfg(feature = "channels-webrtc-advanced")]
    pub fn target_bitrate_bps(&self) -> Option<u32> {
        self.gcc_handle
            .lock()
            .ok()?
            .as_ref()
            .and_then(|h| h.target_bitrate_bps())
    }

    // ── Close ─────────────────────────────────────────────────────────────────

    /// Close the PeerConnection and clean up all resources.
    pub async fn close(&self) -> Result<()> {
        let pc: Option<Arc<dyn PeerConnection>> = self.peer_connection.write().await.take();
        if let Some(pc) = pc {
            pc.close()
                .await
                .map_err(|e| anyhow!("close PeerConnection: {e}"))?;
        }
        self.state.write().await.signaling_state = SignalingState::Closed;
        Ok(())
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    async fn get_pc(&self) -> Result<Arc<dyn PeerConnection>> {
        let guard = self.peer_connection.read().await;
        guard
            .clone()
            .ok_or_else(|| anyhow!("WebRtcSession not opened; call open() first"))
    }
}

// ── Integration tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::super::identity::ConversationId;
    use super::*;
    use std::sync::Arc;

    fn test_conv() -> ConversationId {
        ConversationId {
            platform: "test".to_string(),
            channel_id: "webrtc-lifecycle".to_string(),
            server_id: None,
        }
    }

    /// Two in-process sessions complete a full offer/answer exchange and both
    /// reach `PeerConnectionState::Connected` within 10 seconds.
    ///
    /// Requires `flavor = "multi_thread"` because the webrtc-rs runtime spawns
    /// background tasks that need a real thread pool.
    #[tokio::test(flavor = "multi_thread")]
    async fn offer_answer_reaches_connected() {
        let conv = test_conv();

        let initiator = Arc::new(WebRtcSession::new(WebRtcConfig::default(), conv.clone()));
        let responder = Arc::new(WebRtcSession::new(WebRtcConfig::default(), conv.clone()));

        // Subscribe before open() so the broadcast receiver is registered and
        // we do not miss early ICE candidates buffered in the channel.
        let mut init_rx = initiator.subscribe();
        let mut resp_rx = responder.subscribe();

        initiator.open().await.expect("initiator open");
        responder.open().await.expect("responder open");

        initiator
            .add_audio_track(AudioCodec::Opus)
            .await
            .expect("add audio track");

        // ── Offer / answer exchange ────────────────────────────────────────────
        let offer = initiator.create_offer().await.expect("create offer");
        responder
            .set_remote_description(offer, SdpType::Offer)
            .await
            .expect("set remote description (offer)");
        let answer = responder.create_answer().await.expect("create answer");
        initiator
            .set_remote_description(answer, SdpType::Answer)
            .await
            .expect("set remote description (answer)");

        // ── Bidirectional ICE candidate relay ─────────────────────────────────
        // Each task forwards IceCandidate events to the other peer and exits
        // when PeerConnectionState::Connected (or Failed) is observed.

        let resp_clone = Arc::clone(&responder);
        let init_relay = tokio::spawn(async move {
            loop {
                match init_rx.recv().await {
                    Ok(ChannelEvent::IceCandidate {
                        candidate,
                        sdp_mid,
                        sdp_mline_index,
                        ..
                    }) => {
                        let _ = resp_clone
                            .add_ice_candidate(candidate, sdp_mid, sdp_mline_index)
                            .await;
                    }
                    Ok(ChannelEvent::PeerConnectionStateChanged { state, .. })
                        if state == PeerConnectionState::Connected
                            || state == PeerConnectionState::Failed =>
                    {
                        break;
                    }
                    Err(_) => break,
                    _ => {}
                }
            }
        });

        let init_clone = Arc::clone(&initiator);
        let resp_relay = tokio::spawn(async move {
            loop {
                match resp_rx.recv().await {
                    Ok(ChannelEvent::IceCandidate {
                        candidate,
                        sdp_mid,
                        sdp_mline_index,
                        ..
                    }) => {
                        let _ = init_clone
                            .add_ice_candidate(candidate, sdp_mid, sdp_mline_index)
                            .await;
                    }
                    Ok(ChannelEvent::PeerConnectionStateChanged { state, .. })
                        if state == PeerConnectionState::Connected
                            || state == PeerConnectionState::Failed =>
                    {
                        break;
                    }
                    Err(_) => break,
                    _ => {}
                }
            }
        });

        let (init_res, resp_res) =
            tokio::time::timeout(std::time::Duration::from_secs(10), async {
                tokio::join!(init_relay, resp_relay)
            })
            .await
            .expect("sessions did not connect within 10 seconds");
        init_res.expect("initiator relay task panicked");
        resp_res.expect("responder relay task panicked");

        assert_eq!(
            initiator.peer_connection_state().await,
            PeerConnectionState::Connected,
            "initiator did not reach Connected"
        );
        assert_eq!(
            responder.peer_connection_state().await,
            PeerConnectionState::Connected,
            "responder did not reach Connected"
        );

        initiator.close().await.expect("initiator close");
        responder.close().await.expect("responder close");
    }
}
