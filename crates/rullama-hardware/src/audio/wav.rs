use crate::audio::error::{AudioError, AudioResult};
use crate::audio::types::{AudioBuffer, AudioConfig, SampleFormat};

/// Encode an [`AudioBuffer`] to WAV format bytes.
pub fn encode_wav(buffer: &AudioBuffer) -> AudioResult<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: buffer.config.channels,
        sample_rate: buffer.config.sample_rate,
        bits_per_sample: match buffer.config.sample_format {
            SampleFormat::I16 => 16,
            SampleFormat::F32 => 32,
        },
        sample_format: match buffer.config.sample_format {
            SampleFormat::I16 => hound::SampleFormat::Int,
            SampleFormat::F32 => hound::SampleFormat::Float,
        },
    };

    let mut cursor = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec)
        .map_err(|e| AudioError::Format(format!("WAV encode error: {e}")))?;

    match buffer.config.sample_format {
        SampleFormat::I16 => {
            for chunk in buffer.data.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                writer
                    .write_sample(sample)
                    .map_err(|e| AudioError::Format(format!("WAV write error: {e}")))?;
            }
        }
        SampleFormat::F32 => {
            for chunk in buffer.data.chunks_exact(4) {
                let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                writer
                    .write_sample(sample)
                    .map_err(|e| AudioError::Format(format!("WAV write error: {e}")))?;
            }
        }
    }

    writer
        .finalize()
        .map_err(|e| AudioError::Format(format!("WAV finalize error: {e}")))?;

    Ok(cursor.into_inner())
}

/// Decode WAV format bytes into an [`AudioBuffer`].
pub fn decode_wav(wav_bytes: &[u8]) -> AudioResult<AudioBuffer> {
    let cursor = std::io::Cursor::new(wav_bytes);
    let reader = hound::WavReader::new(cursor)
        .map_err(|e| AudioError::Format(format!("WAV decode error: {e}")))?;

    let spec = reader.spec();
    let sample_format = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => SampleFormat::I16,
        (hound::SampleFormat::Float, 32) => SampleFormat::F32,
        _ => {
            return Err(AudioError::Unsupported(format!(
                "unsupported WAV format: {:?} {}bit",
                spec.sample_format, spec.bits_per_sample
            )));
        }
    };

    let config = AudioConfig {
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        sample_format,
    };

    let data: Vec<u8> = match sample_format {
        SampleFormat::I16 => reader
            .into_samples::<i16>()
            .collect::<Result<Vec<i16>, _>>()
            .map_err(|e| AudioError::Format(format!("WAV sample read error: {e}")))?
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect(),
        SampleFormat::F32 => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()
            .map_err(|e| AudioError::Format(format!("WAV sample read error: {e}")))?
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect(),
    };

    Ok(AudioBuffer { data, config })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_roundtrip_i16() {
        let config = AudioConfig::speech();
        let samples: Vec<i16> = (0..1600).map(|i| (i as i16).wrapping_mul(7)).collect();
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let buffer = AudioBuffer::from_pcm(data.clone(), config);

        let wav_bytes = encode_wav(&buffer).unwrap();
        let decoded = decode_wav(&wav_bytes).unwrap();

        assert_eq!(decoded.config.sample_rate, 16000);
        assert_eq!(decoded.config.channels, 1);
        assert_eq!(decoded.config.sample_format, SampleFormat::I16);
        assert_eq!(decoded.data, data);
    }

    #[test]
    fn test_wav_roundtrip_f32() {
        let config = AudioConfig {
            sample_rate: 48000,
            channels: 2,
            sample_format: SampleFormat::F32,
        };
        let samples: Vec<f32> = (0..960).map(|i| (i as f32) / 960.0).collect();
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let buffer = AudioBuffer::from_pcm(data.clone(), config);

        let wav_bytes = encode_wav(&buffer).unwrap();
        let decoded = decode_wav(&wav_bytes).unwrap();

        assert_eq!(decoded.config.sample_rate, 48000);
        assert_eq!(decoded.config.channels, 2);
        assert_eq!(decoded.config.sample_format, SampleFormat::F32);
        assert_eq!(decoded.data, data);
    }

    #[test]
    fn test_empty_buffer_roundtrip() {
        let config = AudioConfig::speech();
        let buffer = AudioBuffer::new(config);

        let wav_bytes = encode_wav(&buffer).unwrap();
        let decoded = decode_wav(&wav_bytes).unwrap();

        assert!(decoded.is_empty());
    }
}
