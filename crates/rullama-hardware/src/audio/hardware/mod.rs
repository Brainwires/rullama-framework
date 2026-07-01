/// Audio capture backend using cpal.
pub mod cpal_capture;
/// Shared cpal configuration helpers.
pub mod cpal_common;
/// Audio playback backend using cpal.
pub mod cpal_playback;

pub use cpal_capture::CpalCapture;
pub use cpal_playback::CpalPlayback;
