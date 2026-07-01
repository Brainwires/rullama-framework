use async_trait::async_trait;
use futures::stream::BoxStream;
use std::path::PathBuf;

use crate::audio::error::{AudioError, AudioResult};
use crate::audio::stt::SpeechToText;

/// Divisor for normalising i16 PCM samples to the [-1.0, 1.0] range.
/// Equals 2^15 = 32768 (the absolute value of i16::MIN).
const I16_NORMALIZE_DIVISOR: f32 = 32768.0;
use crate::audio::types::{
    AudioBuffer, AudioConfig, SAMPLE_RATE_SPEECH, SampleFormat, SttOptions, Transcript,
    TranscriptSegment,
};

/// Local whisper.cpp speech-to-text implementation via whisper-rs.
pub struct WhisperStt {
    model_path: PathBuf,
}

impl WhisperStt {
    /// Create a new local Whisper STT with the path to a GGML model file.
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
        }
    }

    /// Convert an AudioBuffer to f32 mono 16kHz PCM samples (what whisper.cpp needs).
    fn to_f32_16khz(audio: &AudioBuffer) -> AudioResult<Vec<f32>> {
        let samples_f32: Vec<f32> = match audio.config.sample_format {
            SampleFormat::I16 => audio
                .data
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / I16_NORMALIZE_DIVISOR)
                .collect(),
            SampleFormat::F32 => audio
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        };

        // Mix to mono if stereo
        let mono = if audio.config.channels > 1 {
            let ch = audio.config.channels as usize;
            samples_f32
                .chunks_exact(ch)
                .map(|frame| frame.iter().sum::<f32>() / ch as f32)
                .collect()
        } else {
            samples_f32
        };

        // Simple nearest-neighbor resample to 16kHz if needed
        if audio.config.sample_rate != SAMPLE_RATE_SPEECH {
            let ratio = SAMPLE_RATE_SPEECH as f64 / audio.config.sample_rate as f64;
            let new_len = (mono.len() as f64 * ratio) as usize;
            let resampled: Vec<f32> = (0..new_len)
                .map(|i| {
                    let src_idx = (i as f64 / ratio) as usize;
                    mono.get(src_idx).copied().unwrap_or(0.0)
                })
                .collect();
            Ok(resampled)
        } else {
            Ok(mono)
        }
    }
}

#[async_trait]
impl SpeechToText for WhisperStt {
    fn name(&self) -> &str {
        "whisper-local"
    }

    async fn transcribe(
        &self,
        audio: &AudioBuffer,
        options: &SttOptions,
    ) -> AudioResult<Transcript> {
        let samples = Self::to_f32_16khz(audio)?;
        let model_path = self.model_path.clone();
        let language = options.language.clone();
        let timestamps = options.timestamps;

        // Run inference on a blocking thread since whisper.cpp is CPU-bound
        tokio::task::spawn_blocking(move || {
            let path_str = model_path.to_str().ok_or_else(|| {
                AudioError::Transcription(format!(
                    "model path contains invalid UTF-8: {}",
                    model_path.display()
                ))
            })?;
            let ctx = whisper_rs::WhisperContext::new_with_params(
                path_str,
                whisper_rs::WhisperContextParameters::default(),
            )
            .map_err(|e| AudioError::Transcription(format!("failed to load model: {e}")))?;

            let mut state = ctx
                .create_state()
                .map_err(|e| AudioError::Transcription(format!("failed to create state: {e}")))?;

            let mut params =
                whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });

            if let Some(lang) = &language {
                params.set_language(Some(lang));
            }
            params.set_print_special(false);
            params.set_print_realtime(false);
            params.set_print_progress(false);
            params.set_token_timestamps(timestamps);

            state
                .full(params, &samples)
                .map_err(|e| AudioError::Transcription(format!("inference failed: {e}")))?;

            let num_segments = state.full_n_segments();
            let mut text = String::new();
            let mut segments = Vec::new();

            for i in 0..num_segments {
                let seg = state.get_segment(i).ok_or_else(|| {
                    AudioError::Transcription(format!("segment {i} out of range"))
                })?;
                let segment_text = seg
                    .to_str()
                    .map_err(|e| {
                        AudioError::Transcription(format!("failed to get segment text: {e}"))
                    })?
                    .to_string();
                text.push_str(&segment_text);

                if timestamps {
                    let start = seg.start_timestamp();
                    let end = seg.end_timestamp();
                    segments.push(TranscriptSegment {
                        text: segment_text,
                        start: start as f64 / 100.0, // whisper timestamps are in centiseconds
                        end: end as f64 / 100.0,
                    });
                }
            }

            let duration_secs = Some(samples.len() as f64 / 16000.0);

            Ok::<Transcript, AudioError>(Transcript {
                text,
                language: language.clone(),
                duration_secs,
                segments,
            })
        })
        .await
        .map_err(|e| AudioError::Transcription(format!("task join error: {e}")))?
    }

    fn transcribe_stream(
        &self,
        audio_stream: BoxStream<'static, AudioResult<AudioBuffer>>,
        options: &SttOptions,
    ) -> BoxStream<'static, AudioResult<Transcript>> {
        // For local whisper, buffer all audio then transcribe.
        // Real streaming would require VAD + chunked inference.
        let model_path = self.model_path.clone();
        let options = options.clone();

        let stream = async_stream::stream! {
            use futures::StreamExt;

            let mut all_data = Vec::new();
            let mut config: Option<AudioConfig> = None;
            let mut audio_stream = audio_stream;

            while let Some(result) = audio_stream.next().await {
                match result {
                    Ok(buffer) => {
                        if config.is_none() {
                            config = Some(buffer.config.clone());
                        }
                        all_data.extend_from_slice(&buffer.data);
                    }
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }

            if let Some(cfg) = config {
                let full_buffer = AudioBuffer::from_pcm(all_data, cfg);
                let stt = WhisperStt::new(model_path);
                yield stt.transcribe(&full_buffer, &options).await;
            }
        };

        Box::pin(stream)
    }
}
