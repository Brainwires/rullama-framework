use std::time::Instant;

use async_trait::async_trait;
use nokhwa::{Camera, pixel_format::RgbFormat};
use tracing::debug;

use super::types::{CameraError, CameraFormat, CameraFrame, PixelFormat};

/// Trait for capturing frames from a camera device.
#[async_trait]
pub trait CameraCapture: Send + Sync {
    /// The negotiated capture format (resolution, fps, pixel format).
    fn format(&self) -> CameraFormat;

    /// Capture and return a single frame.
    async fn capture_frame(&mut self) -> Result<CameraFrame, CameraError>;

    /// Stop the capture stream and release the device.
    fn stop(&mut self);
}

/// Cross-platform camera capture backed by [`nokhwa`].
pub struct NokhwaCapture {
    camera: Option<Camera>,
    format: CameraFormat,
    start: Instant,
}

// nokhwa's Camera wraps a C/platform backend that is not Send+Sync by default.
// Access is always single-threaded (capture_frame takes &mut self, stop takes &mut self),
// so it is safe to assert Send+Sync here.
unsafe impl Send for NokhwaCapture {}
unsafe impl Sync for NokhwaCapture {}

impl NokhwaCapture {
    pub(crate) fn new(camera: Camera, format: CameraFormat) -> Self {
        Self {
            camera: Some(camera),
            format,
            start: Instant::now(),
        }
    }
}

#[async_trait]
impl CameraCapture for NokhwaCapture {
    fn format(&self) -> CameraFormat {
        self.format
    }

    async fn capture_frame(&mut self) -> Result<CameraFrame, CameraError> {
        let camera = self
            .camera
            .as_mut()
            .ok_or_else(|| CameraError::CaptureFailed("camera stream has been stopped".into()))?;

        let elapsed_ms = self.start.elapsed().as_millis() as u64;
        let fmt = self.format;

        // nokhwa is synchronous — use block_in_place to avoid blocking the async runtime.
        // block_in_place does not require Send, so the non-Send Camera is fine here.
        tokio::task::block_in_place(|| {
            let buf = camera
                .frame()
                .map_err(|e| CameraError::CaptureFailed(e.to_string()))?;

            // Decode to raw RGB (nokhwa handles all format conversions internally
            // when the RequestedFormat uses RgbFormat)
            let decoded = buf
                .decode_image::<RgbFormat>()
                .map_err(|e| CameraError::CaptureFailed(format!("decode failed: {e}")))?;

            let (width, height) = decoded.dimensions();
            let data = decoded.into_raw();

            debug!(
                "Captured frame {width}x{height} ({} bytes) fmt={fmt:?}",
                data.len()
            );

            Ok(CameraFrame {
                width,
                height,
                format: PixelFormat::Rgb, // always RGB after decode
                data,
                timestamp_ms: elapsed_ms,
            })
        })
    }

    fn stop(&mut self) {
        if let Some(mut cam) = self.camera.take() {
            let _ = cam.stop_stream();
        }
    }
}

impl Drop for NokhwaCapture {
    fn drop(&mut self) {
        self.stop();
    }
}
