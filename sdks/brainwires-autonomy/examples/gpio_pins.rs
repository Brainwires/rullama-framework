//! Example: GPIO Pins — pin management, safety policies, and PWM configuration.
//!
//! ```bash
//! cargo run -p brainwires-autonomy --example gpio_pins --features gpio
//! ```

use brainwires_autonomy::gpio::{
    GpioPinManager, GpioSafetyPolicy,
    config::GpioConfig,
    device::{GpioDirection, discover_chips},
    pwm::PwmConfig,
};

fn main() {
    println!("=== GPIO Pin Management Example ===\n");

    // 1. Default configuration (no pins allowed)
    let default_config = GpioConfig::default();
    println!("--- Default GpioConfig ---");
    println!("  allowed_pins         = {:?}", default_config.allowed_pins);
    println!(
        "  max_concurrent_pins  = {}",
        default_config.max_concurrent_pins
    );
    println!(
        "  auto_release_timeout = {}s",
        default_config.auto_release_timeout_secs
    );
    println!();

    // 2. Safety policy with empty allow-list
    println!("--- Safety Policy (empty allow-list) ---");
    let empty_policy = GpioSafetyPolicy::from_config(&default_config);
    match empty_policy.check(0, 17, "output", "agent-1") {
        Ok(()) => println!("  Pin 0/17: ALLOWED"),
        Err(reason) => println!("  Pin 0/17: BLOCKED — {reason}"),
    }
    println!();

    // 3. Configure with specific pins
    println!("--- Configured Pin Manager ---");
    let config = GpioConfig {
        allowed_pins: vec![(0, 17), (0, 27), (0, 22), (0, 5), (0, 6)],
        max_concurrent_pins: 3,
        auto_release_timeout_secs: 120,
    };

    println!("  Allowed pins: {:?}", config.allowed_pins);
    println!("  Max concurrent: {}", config.max_concurrent_pins);
    println!();

    let mut manager = GpioPinManager::from_config(&config);

    // 4. Acquire pins
    println!("--- Pin Acquisition ---");

    // Acquire allowed pins
    match manager.acquire(0, 17, GpioDirection::Output, "agent-led") {
        Ok(pin) => println!(
            "  Acquired: chip{}/line{} ({}) by {}",
            pin.chip, pin.line, pin.direction, pin.agent_id
        ),
        Err(e) => println!("  Failed: {e}"),
    }

    match manager.acquire(0, 27, GpioDirection::Input, "agent-sensor") {
        Ok(pin) => println!(
            "  Acquired: chip{}/line{} ({}) by {}",
            pin.chip, pin.line, pin.direction, pin.agent_id
        ),
        Err(e) => println!("  Failed: {e}"),
    }

    match manager.acquire(0, 22, GpioDirection::Output, "agent-relay") {
        Ok(pin) => println!(
            "  Acquired: chip{}/line{} ({}) by {}",
            pin.chip, pin.line, pin.direction, pin.agent_id
        ),
        Err(e) => println!("  Failed: {e}"),
    }

    // Exceed concurrent limit
    match manager.acquire(0, 5, GpioDirection::Output, "agent-extra") {
        Ok(pin) => println!("  Acquired: chip{}/line{}", pin.chip, pin.line),
        Err(e) => println!("  Rejected (expected): {e}"),
    }

    // Try disallowed pin
    match manager.acquire(0, 99, GpioDirection::Output, "agent-bad") {
        Ok(_) => println!("  Acquired pin 99 (unexpected)"),
        Err(e) => println!("  Rejected (expected): {e}"),
    }

    // Try double-acquire
    match manager.acquire(0, 17, GpioDirection::Input, "agent-other") {
        Ok(_) => println!("  Double-acquired (unexpected)"),
        Err(e) => println!("  Rejected (expected): {e}"),
    }

    println!("\n  Active pins: {}", manager.active_count());
    println!();

    // 5. List active pins
    println!("--- Active Pins ---");
    for pin in manager.active_pins() {
        println!(
            "  chip{}/line{}: {} (agent: {})",
            pin.chip, pin.line, pin.direction, pin.agent_id
        );
    }
    println!();

    // 6. Release pins
    println!("--- Release Operations ---");
    manager.release(0, 17);
    println!(
        "  Released chip0/line17, active: {}",
        manager.active_count()
    );

    manager.release_agent("agent-relay");
    println!(
        "  Released all pins for agent-relay, active: {}",
        manager.active_count()
    );

    let timed_out = manager.release_timed_out();
    println!(
        "  Timed-out releases: {} (none expected — timeout is 120s)",
        timed_out.len()
    );
    println!();

    // 7. PWM configuration
    println!("--- PWM Configuration ---");

    match PwmConfig::new(1000.0, 0.5) {
        Ok(pwm) => {
            println!("  1kHz @ 50% duty cycle:");
            println!(
                "    period:    {:.3}ms",
                pwm.period().as_secs_f64() * 1000.0
            );
            println!(
                "    high_time: {:.3}ms",
                pwm.high_time().as_secs_f64() * 1000.0
            );
            println!(
                "    low_time:  {:.3}ms",
                pwm.low_time().as_secs_f64() * 1000.0
            );
        }
        Err(e) => println!("  Error: {e}"),
    }

    match PwmConfig::new(50.0, 0.75) {
        Ok(pwm) => {
            println!("  50Hz @ 75% duty cycle (servo):");
            println!(
                "    period:    {:.1}ms",
                pwm.period().as_secs_f64() * 1000.0
            );
            println!(
                "    high_time: {:.1}ms",
                pwm.high_time().as_secs_f64() * 1000.0
            );
        }
        Err(e) => println!("  Error: {e}"),
    }

    // Invalid configs
    match PwmConfig::new(0.0, 0.5) {
        Ok(_) => println!("  0Hz: accepted (unexpected)"),
        Err(e) => println!("  0Hz: rejected — {e}"),
    }
    match PwmConfig::new(1000.0, 1.5) {
        Ok(_) => println!("  150% duty: accepted (unexpected)"),
        Err(e) => println!("  150% duty: rejected — {e}"),
    }
    println!();

    // 8. Chip discovery (will show actual hardware on Linux with GPIO)
    println!("--- GPIO Chip Discovery ---");
    let chips = discover_chips();
    if chips.is_empty() {
        println!("  No GPIO chips found (expected on non-embedded systems)");
    } else {
        for chip in &chips {
            println!(
                "  {} — {} ({} lines)",
                chip.path.display(),
                chip.label,
                chip.num_lines
            );
        }
    }

    println!("\nDone.");
}
