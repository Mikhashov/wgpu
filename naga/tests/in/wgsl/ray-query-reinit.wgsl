enable wgpu_ray_query;

// One query initialized twice: the second initialize must reset the
// initialization tracker, or the second traversal never proceeds.

@group(0) @binding(0)
var first: acceleration_structure;
@group(0) @binding(1)
var second: acceleration_structure;

struct Output {
    first_t: f32,
    second_t: f32,
}

@group(0) @binding(2)
var<storage, read_write> output: Output;

@compute @workgroup_size(1)
fn main() {
    var rq: ray_query;
    let desc = RayDesc(RAY_FLAG_FORCE_OPAQUE, 0xFFu, 0.001, 100.0, vec3<f32>(0.0, 0.0, 5.0), vec3<f32>(0.0, 0.0, -1.0));

    rayQueryInitialize(&rq, first, desc);
    while (rayQueryProceed(&rq)) {}
    output.first_t = rayQueryGetCommittedIntersection(&rq).t;

    rayQueryInitialize(&rq, second, desc);
    while (rayQueryProceed(&rq)) {}
    output.second_t = rayQueryGetCommittedIntersection(&rq).t;
}
