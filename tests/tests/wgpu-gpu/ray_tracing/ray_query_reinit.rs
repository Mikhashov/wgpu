//! A ray query initialized a second time traverses again (the HLSL
//! backend's initialization tracker kept the first traversal's end).

use std::mem;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, BlasBuildEntry, BlasGeometries, BlasGeometrySizeDescriptors,
    BlasTriangleGeometry, BlasTriangleGeometrySizeDescriptor, BufferUsages, CreateBlasDescriptor,
    CreateTlasDescriptor, TlasInstance, VertexFormat,
};
use wgpu_macros::gpu_test;
use wgpu_test::{GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext};

use crate::ray_tracing::acceleration_structure_limits;

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(RAY_QUERY_REINITIALIZE);
}

const SHADER: &str = r#"
enable wgpu_ray_query;
@group(0) @binding(0) var far: acceleration_structure;
@group(0) @binding(1) var near: acceleration_structure;
@group(0) @binding(2) var<storage, read_write> out: array<f32, 2>;

@compute @workgroup_size(1)
fn main() {
    var rq: ray_query;
    let desc = RayDesc(RAY_FLAG_FORCE_OPAQUE, 0xFFu, 0.001, 100.0, vec3<f32>(0.0, 0.0, 5.0), vec3<f32>(0.0, 0.0, -1.0));
    // Nothing of `far` is on the ray: the traversal runs to its end.
    rayQueryInitialize(&rq, far, desc);
    while rayQueryProceed(&rq) {}
    let miss = rayQueryGetCommittedIntersection(&rq);
    out[0] = select(-1.0, miss.t, miss.kind != RAY_QUERY_INTERSECTION_NONE);
    // `near` holds the triangle at z = 0.
    rayQueryInitialize(&rq, near, desc);
    while rayQueryProceed(&rq) {}
    let hit = rayQueryGetCommittedIntersection(&rq);
    out[1] = select(-1.0, hit.t, hit.kind != RAY_QUERY_INTERSECTION_NONE);
}
"#;

#[gpu_test]
static RAY_QUERY_REINITIALIZE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(wgpu::Limits {
                max_acceleration_structures_per_shader_stage: 2,
                ..acceleration_structure_limits()
            })
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
    )
    .run_async(ray_query_reinitialize);

async fn ray_query_reinitialize(ctx: TestingContext) {
    let device = &ctx.device;
    let triangle: [[f32; 3]; 3] = [[-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [0.0, 0.4, 0.0]];
    let vertices = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&triangle),
        usage: BufferUsages::BLAS_INPUT,
    });
    let size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &CreateBlasDescriptor {
            label: None,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![size.clone()],
        },
    );
    // The triangle moved along x: 1000 away from the ray, or on it.
    let top = |x: f32| {
        let mut tlas = device.create_tlas(&CreateTlasDescriptor {
            label: None,
            max_instances: 1,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        });
        tlas[0] = Some(TlasInstance::new(
            &blas,
            [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            0,
            0xFF,
        ));
        tlas
    };
    let (far, near) = (top(1000.0), top(0.0));

    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &size,
                vertex_buffer: &vertices,
                first_vertex: 0,
                vertex_stride: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }],
        [&far, &near],
    );
    ctx.queue.submit([encoder.finish()]);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 8,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 8,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: far.as_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: near.as_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, 8);
    ctx.queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    let hits: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();
    // A miss, then the triangle 5 away.
    assert_eq!(hits[0], -1.0);
    assert!((hits[1] - 5.0).abs() < 1e-3, "second traversal: {hits:?}");
}
