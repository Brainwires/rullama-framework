//! Capture frames from the first available camera and print metadata.
//!
//! Run with:
//! ```bash
//! cargo run -p rullama-hardware --example camera_capture --features camera
//! ```

use rullama_hardware::camera::{self, CameraCapture};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cameras = camera::list_cameras();
    if cameras.is_empty() {
        eprintln!("No cameras found.");
        return Ok(());
    }

    println!("Available cameras ({}):", cameras.len());
    for cam in &cameras {
        println!("  [{}] {} — {:?}", cam.index, cam.name, cam.description);
    }

    println!("\nOpening camera 0 with default format...");
    let mut cap = camera::open_camera(0, None)?;
    let fmt = cap.format();
    println!(
        "Format: {} @ {} ({:?})",
        fmt.resolution, fmt.frame_rate, fmt.pixel_format
    );

    println!("\nCapturing 5 frames:");
    for i in 0..5 {
        match cap.capture_frame().await {
            Ok(frame) => println!(
                "  Frame {i}: {}x{} {} bytes @ {}ms",
                frame.width,
                frame.height,
                frame.data.len(),
                frame.timestamp_ms,
            ),
            Err(e) => eprintln!("  Frame {i} error: {e}"),
        }
    }

    cap.stop();
    println!("Done.");
    Ok(())
}
