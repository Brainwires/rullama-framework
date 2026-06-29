use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    query,
    utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType},
};
use tracing::{debug, warn};

use super::capture::NokhwaCapture;
use super::types::{CameraDevice, CameraError, CameraFormat, FrameRate, PixelFormat, Resolution};

/// Enumerate all available camera and video capture devices.
///
/// Returns an empty list if no cameras are found or the platform camera API
/// is unavailable.
pub fn list_cameras() -> Vec<CameraDevice> {
    match query(ApiBackend::Auto) {
        Ok(cameras) => cameras
            .into_iter()
            .enumerate()
            .map(|(i, info)| {
                debug!("Camera {i}: {}", info.human_name());
                CameraDevice {
                    index: i as u32,
                    name: info.human_name().to_string(),
                    description: Some(info.description().to_string()),
                }
            })
            .collect(),
        Err(e) => {
            warn!("Failed to query cameras: {e}");
            Vec::new()
        }
    }
}

/// Open a camera by index and return a [`NokhwaCapture`] ready to capture frames.
///
/// If `format` is `None`, the first format reported by the device is used.
/// If the requested format is not supported the best available match is chosen.
pub fn open_camera(index: u32, format: Option<CameraFormat>) -> Result<NokhwaCapture, CameraError> {
    let camera_index = CameraIndex::Index(index);

    let requested = match format {
        Some(fmt) => {
            let res = nokhwa::utils::Resolution::new(fmt.resolution.width, fmt.resolution.height);
            let fps = fmt.frame_rate.numerator / fmt.frame_rate.denominator.max(1);
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(
                nokhwa::utils::CameraFormat::new(res, nokhwa::utils::FrameFormat::MJPEG, fps),
            ))
        }
        None => RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    };

    let mut camera = Camera::new(camera_index, requested).map_err(|e| CameraError::OpenFailed {
        index,
        reason: e.to_string(),
    })?;

    camera.open_stream().map_err(|e| CameraError::OpenFailed {
        index,
        reason: format!("stream open failed: {e}"),
    })?;

    // Read back the actual negotiated format
    let actual = camera.camera_format();
    let actual_fmt = CameraFormat::new(
        Resolution::new(actual.resolution().width(), actual.resolution().height()),
        FrameRate::fps(actual.frame_rate()),
        match actual.format() {
            nokhwa::utils::FrameFormat::MJPEG => PixelFormat::Mjpeg,
            nokhwa::utils::FrameFormat::YUYV => PixelFormat::Yuv422,
            nokhwa::utils::FrameFormat::RAWRGB => PixelFormat::Rgb,
            _ => PixelFormat::Unknown,
        },
    );

    debug!(
        "Opened camera {index}: {}x{} @ {} fps ({:?})",
        actual_fmt.resolution.width,
        actual_fmt.resolution.height,
        actual_fmt.frame_rate.as_f64(),
        actual_fmt.pixel_format
    );

    Ok(NokhwaCapture::new(camera, actual_fmt))
}
