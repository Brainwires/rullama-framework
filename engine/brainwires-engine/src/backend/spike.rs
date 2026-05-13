//! M0 compute spike: dispatch a trivial WGSL kernel that doubles each f32, read the
//! result back. Validates wgpu device → pipeline → buffer → dispatch → readback on the
//! current platform (native or browser).
//!
//! Not part of the production inference path. Keep until M2 lands real kernels, then
//! retain as the simplest end-to-end smoke test.

use crate::backend::WgpuCtx;
use crate::error::{Result, RullamaError};

const SPIKE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       inp:  array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&inp)) { return; }
    outp[i] = inp[i] * 2.0;
}
"#;

/// Run the doubling kernel on `input` and return the output. Used in tests.
pub async fn compute_spike(input: &[f32]) -> Result<Vec<f32>> {
    let ctx = WgpuCtx::new().await?;
    run(&ctx, input).await
}

async fn run(ctx: &WgpuCtx, input: &[f32]) -> Result<Vec<f32>> {
    let device = &ctx.device;
    let queue = &ctx.queue;

    let bytes: u64 = (input.len() * core::mem::size_of::<f32>()) as u64;

    let inp_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spike.in"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&inp_buf, 0, bytemuck::cast_slice(input));

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spike.out"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spike.read"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("spike.wgsl"),
        source: wgpu::ShaderSource::Wgsl(SPIKE_WGSL.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("spike.pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("spike.bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: inp_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("spike.encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("spike.pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = input.len().div_ceil(64) as u32;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, bytes);
    queue.submit(Some(encoder.finish()));

    let slice = read_buf.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .map_err(|e| RullamaError::Inference(format!("{e:?}")))?;
    receiver
        .await
        .map_err(|e| RullamaError::BufferMap(format!("{e}")))?
        .map_err(|e| RullamaError::BufferMap(format!("{e}")))?;

    let data = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    read_buf.unmap();
    Ok(out)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn doubles_a_buffer() {
        let _ = env_logger::builder().is_test(true).try_init();
        let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let output = pollster::block_on(compute_spike(&input)).expect("spike");
        for (i, v) in output.iter().enumerate() {
            assert!((*v - input[i] * 2.0).abs() < 1e-6, "i={i} got={v}");
        }
    }
}
