//! A top level built again after a few of its instances changed (wgpu-core
//! uploads only those) traces like one built from every instance.

use std::mem;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, Blas, BlasBuildEntry, BlasGeometries,
    BlasGeometrySizeDescriptors, BlasTriangleGeometry, BlasTriangleGeometrySizeDescriptor,
    BufferUsages, CreateBlasDescriptor, CreateTlasDescriptor, Tlas, TlasInstance, VertexFormat,
};
use wgpu_macros::gpu_test;
use wgpu_test::{GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext};

use crate::ray_tracing::acceleration_structure_limits;

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(TLAS_PARTIAL_UPLOAD);
}

/// Instance slots, and places along x a ray looks at (2 apart).
const SLOTS: usize = 64;
const PLACES: usize = 128;

const SHADER: &str = r#"
enable wgpu_ray_query;
@group(0) @binding(0) var acc: acceleration_structure;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
// One ray down each place: the custom data of the instance hit, plus one.
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var rq: ray_query;
    let origin = vec3<f32>(f32(id.x) * 2.0, 0.0, 5.0);
    rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, 0.01, 100.0, origin, vec3<f32>(0.0, 0.0, -1.0)));
    while rayQueryProceed(&rq) {}
    let hit = rayQueryGetCommittedIntersection(&rq);
    out[id.x] = select(0u, hit.instance_custom_data + 1u, hit.kind != RAY_QUERY_INTERSECTION_NONE);
}
"#;

fn instance(blas: &Blas, place: usize, custom_data: u32) -> Option<TlasInstance> {
    let x = place as f32 * 2.0;
    Some(TlasInstance::new(
        blas,
        [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        custom_data,
        0xFF,
    ))
}

#[gpu_test]
static TLAS_PARTIAL_UPLOAD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
    )
    .run_async(tlas_partial_upload);

async fn tlas_partial_upload(ctx: TestingContext) {
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
    let bottom = || {
        device.create_blas(
            &CreateBlasDescriptor {
                label: None,
                flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: AccelerationStructureUpdateMode::Build,
            },
            BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size.clone()],
            },
        )
    };
    let (blas, other_blas) = (bottom(), bottom());
    let entry = |blas| BlasBuildEntry {
        blas,
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
    };

    let mut tlas = device.create_tlas(&CreateTlasDescriptor {
        label: None,
        max_instances: SLOTS as u32,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    });

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
        size: PLACES as u64 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: PLACES as u64 * 4,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Builds `tlas` (with the bottom levels the first time) and returns what the rays hit.
    let trace = |tlas: &Tlas, bottoms: bool| {
        let mut encoder = device.create_command_encoder(&Default::default());
        let entries = [entry(&blas), entry(&other_blas)];
        encoder.build_acceleration_structures(entries.iter().take(2 * bottoms as usize), [tlas]);
        ctx.queue.submit([encoder.finish()]);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tlas.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(PLACES as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, PLACES as u64 * 4);
        ctx.queue.submit([encoder.finish()]);
        let ctx = &ctx;
        let readback = &readback;
        async move {
            let slice = readback.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| ());
            ctx.async_poll(wgpu::PollType::wait_indefinitely())
                .await
                .unwrap();
            let hits: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();
            readback.unmap();
            hits
        }
    };
    // What the rays must hit: each slot's place and custom data.
    let expected = |slots: &[Option<(usize, u32)>]| {
        let mut hits = vec![0u32; PLACES];
        for &(place, custom_data) in slots.iter().flatten() {
            hits[place] = custom_data + 1;
        }
        hits
    };

    // Every slot but 3, each at its own place.
    let mut slots: Vec<Option<(usize, u32)>> = (0..SLOTS).map(|k| Some((k, k as u32))).collect();
    slots[3] = None;
    for (k, slot) in slots.iter().enumerate() {
        tlas[k] = slot.and_then(|(place, custom_data)| instance(&blas, place, custom_data));
    }
    assert_eq!(trace(&tlas, true).await, expected(&slots), "first build");

    // One instance moves.
    slots[10] = Some((100, 10));
    tlas[10] = instance(&blas, 100, 10);
    assert_eq!(trace(&tlas, false).await, expected(&slots), "a move");

    // Nothing changes.
    assert_eq!(trace(&tlas, false).await, expected(&slots), "no change");

    // An empty slot is taken and a taken one emptied: the instances after
    // them move in the instance buffer.
    slots[3] = Some((3, 3));
    tlas[3] = instance(&blas, 3, 3);
    slots[20] = None;
    tlas[20] = None;
    assert_eq!(trace(&tlas, false).await, expected(&slots), "slots filled and emptied");

    // Another bottom level and custom data, and the last slot.
    slots[5] = Some((5, 500));
    tlas[5] = instance(&other_blas, 5, 500);
    slots[SLOTS - 1] = Some((120, 7));
    tlas[SLOTS - 1] = instance(&blas, 120, 7);
    assert_eq!(trace(&tlas, false).await, expected(&slots), "another bottom level");

    // A clone has its own instances: building it does not leave the
    // original's next build with the clone's.
    let mut clone = tlas.clone();
    clone[0] = None;
    let mut clone_slots = slots.clone();
    clone_slots[0] = None;
    assert_eq!(trace(&clone, false).await, expected(&clone_slots), "a clone");
    slots[1] = Some((110, 1));
    tlas[1] = instance(&blas, 110, 1);
    assert_eq!(trace(&tlas, false).await, expected(&slots), "the original after its clone");

    // A build recorded and never submitted leaves the next one whole.
    slots[2] = Some((90, 2));
    tlas[2] = instance(&blas, 90, 2);
    let mut dropped = device.create_command_encoder(&Default::default());
    dropped.build_acceleration_structures([], [&tlas]);
    drop(dropped);
    slots[4] = Some((92, 4));
    tlas[4] = instance(&blas, 92, 4);
    assert_eq!(trace(&tlas, false).await, expected(&slots), "after a dropped build");
}
