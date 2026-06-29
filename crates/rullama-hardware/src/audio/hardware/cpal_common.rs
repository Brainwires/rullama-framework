use cpal::traits::{DeviceTrait, HostTrait};

use crate::audio::device::{AudioDevice, DeviceDirection};
use crate::audio::error::{AudioError, AudioResult};
use crate::audio::types::{AudioConfig, SampleFormat};

/// Get the default cpal host.
pub fn default_host() -> cpal::Host {
    cpal::default_host()
}

/// List all input devices.
pub fn list_input_devices() -> AudioResult<Vec<AudioDevice>> {
    let host = default_host();
    let default_name = host.default_input_device().and_then(|d| d.name().ok());

    let devices = host
        .input_devices()
        .map_err(|e| AudioError::Device(format!("failed to enumerate input devices: {e}")))?;

    let mut result = Vec::new();
    for device in devices {
        let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        let is_default = default_name.as_deref() == Some(&name);
        result.push(AudioDevice {
            id: name.clone(),
            name,
            is_default,
            direction: DeviceDirection::Input,
        });
    }
    Ok(result)
}

/// List all output devices.
pub fn list_output_devices() -> AudioResult<Vec<AudioDevice>> {
    let host = default_host();
    let default_name = host.default_output_device().and_then(|d| d.name().ok());

    let devices = host
        .output_devices()
        .map_err(|e| AudioError::Device(format!("failed to enumerate output devices: {e}")))?;

    let mut result = Vec::new();
    for device in devices {
        let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        let is_default = default_name.as_deref() == Some(&name);
        result.push(AudioDevice {
            id: name.clone(),
            name,
            is_default,
            direction: DeviceDirection::Output,
        });
    }
    Ok(result)
}

/// Find a cpal input device by name, or return the default.
pub fn find_input_device(device: Option<&AudioDevice>) -> AudioResult<cpal::Device> {
    let host = default_host();
    match device {
        Some(dev) => {
            let devices = host
                .input_devices()
                .map_err(|e| AudioError::Device(format!("failed to enumerate devices: {e}")))?;
            for d in devices {
                if d.name().ok().as_deref() == Some(&dev.id) {
                    return Ok(d);
                }
            }
            Err(AudioError::Device(format!(
                "input device not found: {}",
                dev.id
            )))
        }
        None => host
            .default_input_device()
            .ok_or_else(|| AudioError::Device("no default input device".to_string())),
    }
}

/// Find a cpal output device by name, or return the default.
pub fn find_output_device(device: Option<&AudioDevice>) -> AudioResult<cpal::Device> {
    let host = default_host();
    match device {
        Some(dev) => {
            let devices = host
                .output_devices()
                .map_err(|e| AudioError::Device(format!("failed to enumerate devices: {e}")))?;
            for d in devices {
                if d.name().ok().as_deref() == Some(&dev.id) {
                    return Ok(d);
                }
            }
            Err(AudioError::Device(format!(
                "output device not found: {}",
                dev.id
            )))
        }
        None => host
            .default_output_device()
            .ok_or_else(|| AudioError::Device("no default output device".to_string())),
    }
}

/// Build a cpal stream config from an [`AudioConfig`].
pub fn build_stream_config(config: &AudioConfig) -> cpal::StreamConfig {
    cpal::StreamConfig {
        channels: config.channels,
        sample_rate: cpal::SampleRate(config.sample_rate),
        buffer_size: cpal::BufferSize::Default,
    }
}

/// Convert our [`SampleFormat`] to cpal's.
pub fn to_cpal_sample_format(format: SampleFormat) -> cpal::SampleFormat {
    match format {
        SampleFormat::I16 => cpal::SampleFormat::I16,
        SampleFormat::F32 => cpal::SampleFormat::F32,
    }
}
