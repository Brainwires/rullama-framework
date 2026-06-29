//! Flow-matching Euler scheduler for the image diffusion sampling loop.
//!
//! Direct port of Ollama's `x/imagegen/models/zimage/scheduler.go`
//! (`FlowMatchEulerScheduler`). Flow matching integrates from t=1 (pure noise)
//! to t=0 (clean) with a velocity-predicting model:
//!
//!   sigmas[i] = timeShift(mu, 1 - i/num_steps)          for i in 0..=num_steps
//!   x_{i+1}   = x_i + (sigmas[i+1] - sigmas[i]) * v_i    (Euler step, dt < 0)
//!
//! The static `shift` in the config isn't applied in `SetTimesteps` itself
//! (it's plain linspace); dynamic shifting folds resolution-dependent `mu`
//! into `timeShift`. `mu == 0` (or non-dynamic) ⇒ identity, i.e. raw linspace.
//!
//! Pure host-side math: the per-step model forward runs on the GPU, the FMA
//! between steps is cheap and done here. No model weights needed for the
//! schedule itself.

/// A built sigma/timestep schedule + the flow-match Euler update.
#[derive(Debug, Clone)]
pub struct FlowMatchScheduler {
    /// `num_steps + 1` sigmas, sigmas[0] = 1 (noise) … sigmas[num_steps] = 0.
    pub sigmas: Vec<f32>,
    pub num_steps: usize,
}

impl FlowMatchScheduler {
    /// Build the schedule. `mu` drives dynamic shifting (pass 0 for plain
    /// linspace, matching `SetTimesteps`). `dynamic == false` also disables it.
    pub fn new(num_steps: usize, dynamic: bool, mu: f32) -> Self {
        let mut sigmas = Vec::with_capacity(num_steps + 1);
        for i in 0..=num_steps {
            let t = 1.0 - (i as f32) / (num_steps as f32);
            let t = if dynamic && mu != 0.0 {
                time_shift(mu, t)
            } else {
                t
            };
            sigmas.push(t);
        }
        Self { sigmas, num_steps }
    }

    /// Sigma at step index `i`.
    pub fn sigma(&self, i: usize) -> f32 {
        self.sigmas[i]
    }

    /// Euler step in place: `x += (sigma_{i+1} - sigma_i) * velocity`.
    /// `velocity` is the model's predicted velocity for the current sample.
    pub fn step_in_place(&self, x: &mut [f32], velocity: &[f32], i: usize) {
        assert_eq!(x.len(), velocity.len());
        let dt = self.sigmas[i + 1] - self.sigmas[i]; // negative
        for (xi, &v) in x.iter_mut().zip(velocity) {
            *xi += dt * v;
        }
    }

    /// Flow-match noising: `x_t = (1 - t) * clean + t * noise`, `t = sigma_i`.
    pub fn add_noise(&self, clean: &[f32], noise: &[f32], i: usize) -> Vec<f32> {
        assert_eq!(clean.len(), noise.len());
        let t = self.sigmas[i];
        let one_minus_t = 1.0 - t;
        clean
            .iter()
            .zip(noise)
            .map(|(&c, &n)| c * one_minus_t + n * t)
            .collect()
    }
}

/// Resolution-dependent shift `mu`, linearly interpolated from the image token
/// count (`latent_h/patch * latent_w/patch`). Mirrors `zimage.go`'s
/// `CalculateShift`: `mu = m*seq + b` over `[256→0.5, 4096→1.15]`.
///
/// NOTE: Z-Image's `scheduler_config.json` says `use_dynamic_shifting=false`,
/// but Ollama (our oracle) ignores that and ALWAYS applies this dynamic `mu` —
/// so the live schedule is `FlowMatchScheduler::new(steps, true, calculate_shift(seq))`.
pub fn calculate_shift(img_seq_len: usize) -> f32 {
    let base_seq = 256.0f32;
    let max_seq = 4096.0f32;
    let base_shift = 0.5f32;
    let max_shift = 1.15f32;
    let m = (max_shift - base_shift) / (max_seq - base_seq);
    let b = base_shift - m * base_seq;
    (img_seq_len as f32) * m + b
}

/// Dynamic time shift: `exp(mu) / (exp(mu) + (1/t - 1))`, with `t<=0 → 0`.
/// Mirrors `scheduler.go`'s `timeShift`.
pub fn time_shift(mu: f32, t: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    let exp_mu = mu.exp();
    exp_mu / (exp_mu + (1.0 / t - 1.0))
}

/// Latent spatial shape for a target image size, given the VAE downscale (8×).
/// Returns `(latent_h, latent_w)`. Mirrors `GetLatentShape`.
pub fn latent_hw(height: usize, width: usize, vae_downscale: usize) -> (usize, usize) {
    (height / vae_downscale, width / vae_downscale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_endpoints_and_monotonic() {
        let s = FlowMatchScheduler::new(8, false, 0.0);
        assert_eq!(s.sigmas.len(), 9);
        assert!((s.sigmas[0] - 1.0).abs() < 1e-6);
        assert!(s.sigmas[8].abs() < 1e-6);
        // strictly decreasing
        for w in s.sigmas.windows(2) {
            assert!(w[1] < w[0], "non-monotonic: {w:?}");
        }
        // even spacing for plain linspace
        assert!((s.sigmas[4] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mu_zero_is_identity() {
        // dynamic with mu=0 ⇒ raw linspace (matches SetTimesteps default).
        let a = FlowMatchScheduler::new(4, true, 0.0);
        let b = FlowMatchScheduler::new(4, false, 0.0);
        assert_eq!(a.sigmas, b.sigmas);
    }

    #[test]
    fn time_shift_known_values() {
        // t=0 → 0; and shift pulls the midpoint toward 1 for mu>0.
        assert_eq!(time_shift(1.5, 0.0), 0.0);
        // exp(0)=1 → 1/(1 + (1/t - 1)) = t  (mu=0 is identity)
        assert!((time_shift(0.0, 0.3) - 0.3).abs() < 1e-6);
        // hand-computed: mu=ln(2) ⇒ expMu=2, t=0.5 ⇒ 2/(2 + (2-1)) = 2/3
        assert!((time_shift(2f32.ln(), 0.5) - (2.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn euler_step_moves_toward_clean() {
        let s = FlowMatchScheduler::new(2, false, 0.0); // sigmas 1, 0.5, 0
        let mut x = vec![10.0f32, -4.0];
        let v = vec![2.0f32, 1.0];
        // step 0: dt = 0.5 - 1.0 = -0.5 → x += -0.5 * v
        s.step_in_place(&mut x, &v, 0);
        assert_eq!(x, vec![10.0 - 1.0, -4.0 - 0.5]);
    }

    #[test]
    fn add_noise_blends() {
        let s = FlowMatchScheduler::new(2, false, 0.0); // sigmas 1, 0.5, 0
        // at i=1, t=0.5 → 0.5*clean + 0.5*noise
        let out = s.add_noise(&[2.0, 4.0], &[0.0, 0.0], 1);
        assert_eq!(out, vec![1.0, 2.0]);
        // at i=2, t=0 → all clean
        assert_eq!(s.add_noise(&[2.0, 4.0], &[9.0, 9.0], 2), vec![2.0, 4.0]);
    }

    #[test]
    fn latent_shape_divides_by_downscale() {
        assert_eq!(latent_hw(1024, 768, 8), (128, 96));
    }

    #[test]
    fn calculate_shift_endpoints() {
        // Linear in seq len: 256 → 0.5, 4096 → 1.15.
        assert!((calculate_shift(256) - 0.5).abs() < 1e-5);
        assert!((calculate_shift(4096) - 1.15).abs() < 1e-5);
        // midpoint of the range is the midpoint shift
        let mid = calculate_shift((256 + 4096) / 2);
        assert!((mid - (0.5 + 1.15) / 2.0).abs() < 1e-4);
    }

    #[test]
    fn dynamic_schedule_with_calculated_mu_still_spans_1_to_0() {
        // A realistic 1024² image: 128×128 latent, patch 2 → 64×64 = 4096 tokens.
        let mu = calculate_shift(64 * 64);
        let s = FlowMatchScheduler::new(9, true, mu);
        assert!((s.sigmas[0] - 1.0).abs() < 1e-5);
        assert!(s.sigmas[9].abs() < 1e-5);
        for w in s.sigmas.windows(2) {
            assert!(w[1] < w[0]);
        }
    }
}
