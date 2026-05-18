fn main() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .unwrap();
    let info = adapter.get_info();
    println!("backend: {:?}", info.backend);
    println!("name: {}", info.name);
    println!("device_type: {:?}", info.device_type);
    println!("vendor: 0x{:x}", info.vendor);
    println!("driver: {}", info.driver);
    println!("driver_info: {}", info.driver_info);

    let feats = adapter.features();
    let limits = adapter.limits();
    println!("\nFeature support:");
    for (name, bit) in &[
        ("SUBGROUP", wgpu::Features::SUBGROUP),
        ("SUBGROUP_BARRIER", wgpu::Features::SUBGROUP_BARRIER),
        ("SUBGROUP_VERTEX", wgpu::Features::SUBGROUP_VERTEX),
        ("SHADER_F16", wgpu::Features::SHADER_F16),
        ("TIMESTAMP_QUERY", wgpu::Features::TIMESTAMP_QUERY),
        (
            "TIMESTAMP_QUERY_INSIDE_PASSES",
            wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        ),
        (
            "PIPELINE_STATISTICS_QUERY",
            wgpu::Features::PIPELINE_STATISTICS_QUERY,
        ),
    ] {
        println!("  {:32} = {}", name, feats.contains(*bit));
    }

    println!("\nKey limits:");
    println!(
        "  max_compute_workgroup_storage_size      = {}",
        limits.max_compute_workgroup_storage_size
    );
    println!(
        "  max_compute_invocations_per_workgroup   = {}",
        limits.max_compute_invocations_per_workgroup
    );
    println!(
        "  max_compute_workgroup_size_x            = {}",
        limits.max_compute_workgroup_size_x
    );
    println!(
        "  max_storage_buffer_binding_size         = {}",
        limits.max_storage_buffer_binding_size
    );
    println!(
        "  max_buffer_size                         = {}",
        limits.max_buffer_size
    );
    println!("\nSubgroup adapter info:");
    println!(
        "  subgroup_min_size                       = {}",
        info.subgroup_min_size
    );
    println!(
        "  subgroup_max_size                       = {}",
        info.subgroup_max_size
    );
}
