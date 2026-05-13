//! Learning rate scheduling for local training.
//!
//! Implements warmup + decay schedules matching `LrScheduler` config variants.

use crate::shared::config::LrScheduler;

/// A learning rate schedule that computes LR for each training step.
pub struct LrSchedule {
    base_lr: f64,
    warmup_steps: u64,
    total_steps: u64,
    scheduler_type: LrScheduler,
}

impl LrSchedule {
    /// Create a new LR schedule.
    pub fn new(
        base_lr: f64,
        warmup_steps: u64,
        total_steps: u64,
        scheduler_type: LrScheduler,
    ) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            scheduler_type,
        }
    }

    /// Get the learning rate at the given step.
    pub fn get_lr(&self, step: u64) -> f64 {
        if step == 0 {
            return 0.0;
        }

        // Warmup phase: linear ramp from 0 to base_lr
        if step <= self.warmup_steps {
            return self.base_lr * (step as f64 / self.warmup_steps.max(1) as f64);
        }

        // Decay phase
        let decay_step = step - self.warmup_steps;
        let decay_total = self.total_steps.saturating_sub(self.warmup_steps).max(1);

        match self.scheduler_type {
            LrScheduler::Constant => self.base_lr,
            LrScheduler::Linear => {
                let progress = decay_step as f64 / decay_total as f64;
                self.base_lr * (1.0 - progress).max(0.0)
            }
            LrScheduler::Cosine => {
                // Clamp at 1.0 so an off-by-one past `total_steps` doesn't swing
                // cos(π·progress) back positive and re-raise the LR.
                let progress = (decay_step as f64 / decay_total as f64).min(1.0);
                self.base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
            }
            LrScheduler::CosineWarmRestarts => {
                // T_0 = decay_total / 2, restart once
                let t_0 = (decay_total as f64 / 2.0).max(1.0);
                let t_cur = decay_step as f64 % t_0;
                self.base_lr * 0.5 * (1.0 + (std::f64::consts::PI * t_cur / t_0).cos())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_schedule() {
        let sched = LrSchedule::new(1e-4, 10, 100, LrScheduler::Constant);
        // Warmup
        assert!((sched.get_lr(5) - 5e-5).abs() < 1e-10);
        // After warmup: constant
        assert!((sched.get_lr(10) - 1e-4).abs() < 1e-10);
        assert!((sched.get_lr(50) - 1e-4).abs() < 1e-10);
        assert!((sched.get_lr(100) - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_linear_schedule() {
        let sched = LrSchedule::new(1e-4, 0, 100, LrScheduler::Linear);
        // Should decay linearly to 0
        assert!((sched.get_lr(1) - 1e-4 * 0.99).abs() < 1e-10);
        assert!((sched.get_lr(50) - 1e-4 * 0.5).abs() < 1e-10);
        assert!(sched.get_lr(100) < 1e-10);
    }

    #[test]
    fn test_cosine_schedule() {
        let sched = LrSchedule::new(1e-4, 10, 110, LrScheduler::Cosine);
        // After warmup, should follow cosine
        let lr_mid = sched.get_lr(60); // halfway through decay
        assert!((lr_mid - 1e-4 * 0.5).abs() < 1e-6);
        // End should be near 0
        assert!(sched.get_lr(110) < 1e-8);
    }

    #[test]
    fn test_warmup_ramp() {
        let sched = LrSchedule::new(1e-3, 100, 1000, LrScheduler::Cosine);
        assert_eq!(sched.get_lr(0), 0.0);
        assert!((sched.get_lr(50) - 5e-4).abs() < 1e-10);
        assert!((sched.get_lr(100) - 1e-3).abs() < 1e-10);
    }
}
