//! Every command of `multi_draw_indexed_indirect_count` gets its own
//! `first_instance` and `base_vertex`: in `instance_index` and `vertex_index`
//! (D3D12 takes those from special constants, which the indirect validation
//! pass writes), and in what instance-rate and per-vertex buffers are read
//! at (D3D12 does that natively, so such a pipeline keeps the native path).

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend(&[
        MULTI_DRAW_INDEXED_INDIRECT_COUNT_BUILTINS,
        MULTI_DRAW_INDEXED_INDIRECT_COUNT_BUILTINS_LARGE,
        MULTI_DRAW_INDEXED_INDIRECT_COUNT_BUFFERS,
        MULTI_DRAW_INDEXED_INDIRECT_COUNT_BUFFERS_LARGE,
    ]);
}

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> seen: array<atomic<u32>>;
struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) ids: vec2<u32>,
};
// A triangle over the whole target per command; the fragments record the
// first vertex's index (plus one) under the instance's.
@vertex fn vs(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> Out {
    let corner = vec2<f32>(f32((v % 10u) & 1u), f32((v % 10u) >> 1u)) * 4.0 - 1.0;
    return Out(vec4<f32>(corner, 0.5, 1.0), vec2<u32>(i, v));
}
// The same from an instance-rate and a per-vertex buffer of 0, 1, 2, ...: no
// built-in is read.
@vertex fn vs_buffers(@location(0) i: u32, @location(1) v: u32) -> Out {
    let corner = vec2<f32>(f32((v % 10u) & 1u), f32((v % 10u) >> 1u)) * 4.0 - 1.0;
    return Out(vec4<f32>(corner, 0.5, 1.0), vec2<u32>(i, v));
}
@fragment fn fs(in: Out) -> @location(0) vec4<f32> {
    atomicMax(&seen[in.ids.x], in.ids.y + 1u);
    return vec4<f32>(0.0);
}
"#;

/// Draws `max_count` commands, the last four of which draw one instance each
/// (first instance 5, 8, 11, 14; base vertex 0, 10, 20, 30); the rest are
/// empty. With `buffers` the instance and the vertex come from vertex
/// buffers instead of the built-ins.
async fn run(ctx: TestingContext, max_count: u32, buffers: bool) {
    let device = &ctx.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some(if buffers { "vs_buffers" } else { "vs" }),
            compilation_options: Default::default(),
            buffers: if buffers {
                &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 4,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Uint32],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 4,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![1 => Uint32],
                    }),
                ]
            } else {
                &[]
            },
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
        }),
        multiview_mask: None,
        cache: None,
    });
    let init = |contents: &[u8], usage| {
        device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents,
            usage,
        })
    };
    let seen = init(
        &[0u8; 64 * 4],
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let indices = init(
        bytemuck::cast_slice(&[0u32, 1, 2]),
        wgpu::BufferUsages::INDEX,
    );
    let identity: Vec<u32> = (0..64).collect();
    let stream = init(bytemuck::cast_slice(&identity), wgpu::BufferUsages::VERTEX);
    let mut commands = vec![0u32; 5 * (max_count as usize - 4)];
    commands.extend((0..4u32).flat_map(|k| [3, 1, 0, 10 * k, 5 + 3 * k]));
    let commands = init(
        bytemuck::cast_slice(&commands),
        wgpu::BufferUsages::INDIRECT,
    );
    let count = init(
        bytemuck::cast_slice(&[max_count]),
        wgpu::BufferUsages::INDIRECT,
    );
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: seen.as_entire_binding(),
        }],
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: Default::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
        if buffers {
            pass.set_vertex_buffer(0, stream.slice(..));
            pass.set_vertex_buffer(1, stream.slice(..));
        }
        pass.multi_draw_indexed_indirect_count(&commands, 0, &count, 0, max_count);
    }
    encoder.copy_buffer_to_buffer(&seen, 0, &readback, 0, 64 * 4);
    ctx.queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    let seen: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();

    // Instance 5 + 3k saw its first vertex, 10k.
    let expected: Vec<u32> = (0..64u32)
        .map(|i| {
            if (5..=14).contains(&i) && (i - 5) % 3 == 0 {
                10 * ((i - 5) / 3) + 1
            } else {
                0
            }
        })
        .collect();
    assert_eq!(seen, expected);
}

fn parameters() -> TestParameters {
    TestParameters::default()
        .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
        .features(
            wgpu::Features::MULTI_DRAW_INDIRECT_COUNT | wgpu::Features::INDIRECT_FIRST_INSTANCE,
        )
        .limits(wgpu::Limits::downlevel_defaults())
}

#[gpu_test]
static MULTI_DRAW_INDEXED_INDIRECT_COUNT_BUILTINS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(parameters())
        .run_async(|ctx| run(ctx, 4, false));

/// More commands than one buffer of the indirect validation holds on D3D12.
#[gpu_test]
static MULTI_DRAW_INDEXED_INDIRECT_COUNT_BUILTINS_LARGE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(parameters())
        .run_async(|ctx| run(ctx, 40_000, false));

/// A vertex stage that reads no built-in: the commands' first instance and
/// base vertex reach the vertex buffers (on D3D12 without the validation pass).
#[gpu_test]
static MULTI_DRAW_INDEXED_INDIRECT_COUNT_BUFFERS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(parameters())
        .run_async(|ctx| run(ctx, 4, true));

#[gpu_test]
static MULTI_DRAW_INDEXED_INDIRECT_COUNT_BUFFERS_LARGE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(parameters())
        .run_async(|ctx| run(ctx, 40_000, true));
