struct Output {
    float first_t;
    float second_t;
};

struct RayDesc_ {
    uint flags;
    uint cull_mask;
    float tmin;
    float tmax;
    float3 origin;
    int _pad5_0;
    float3 dir;
    int _end_pad_0;
};

struct RayIntersection {
    uint kind;
    float t;
    uint instance_custom_data;
    uint instance_index;
    uint sbt_record_offset;
    uint geometry_index;
    uint primitive_index;
    float2 barycentrics;
    bool front_face;
    int _pad9_0;
    int _pad9_1;
    row_major float4x3 object_to_world;
    int _pad10_0;
    row_major float4x3 world_to_object;
    int _end_pad_0;
};

RayDesc RayDescFromRayDesc_(RayDesc_ arg0) {
    RayDesc ret = (RayDesc)0;
    ret.Origin = arg0.origin;
    ret.TMin = arg0.tmin;
    ret.Direction = arg0.dir;
    ret.TMax = arg0.tmax;
    return ret;
}

RaytracingAccelerationStructure first : register(t0);
RaytracingAccelerationStructure second : register(t1);
RWByteAddressBuffer output : register(u2);

RayDesc_ ConstructRayDesc_(uint arg0, uint arg1, float arg2, float arg3, float3 arg4, float3 arg5) {
    RayDesc_ ret = (RayDesc_)0;
    ret.flags = arg0;
    ret.cull_mask = arg1;
    ret.tmin = arg2;
    ret.tmax = arg3;
    ret.origin = arg4;
    ret.dir = arg5;
    return ret;
}

RayIntersection GetCommittedIntersection(RayQuery<RAY_FLAG_NONE> rq, uint rq_tracker) {
    RayIntersection ret = (RayIntersection)0;
    if (((rq_tracker & 4) == 4)) {
        ret.kind = rq.CommittedStatus();
        if( rq.CommittedStatus() == COMMITTED_NOTHING) {} else {
            ret.t = rq.CommittedRayT();
            ret.instance_custom_data = rq.CommittedInstanceID();
            ret.instance_index = rq.CommittedInstanceIndex();
            ret.sbt_record_offset = rq.CommittedInstanceContributionToHitGroupIndex();
            ret.geometry_index = rq.CommittedGeometryIndex();
            ret.primitive_index = rq.CommittedPrimitiveIndex();
            if( rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT ) {
                ret.barycentrics = rq.CommittedTriangleBarycentrics();
                ret.front_face = rq.CommittedTriangleFrontFace();
            }
            ret.object_to_world = rq.CommittedObjectToWorld4x3();
            ret.world_to_object = rq.CommittedWorldToObject4x3();
        }
    }
    return ret;
}

[numthreads(1, 1, 1)]
void main()
{
    RayQuery<RAY_FLAG_NONE> rq;
    uint naga_query_init_tracker_for_rq = 0;

    RayDesc_ desc = ConstructRayDesc_(1u, 255u, 0.001, 100.0, float3(0.0, 0.0, 5.0), float3(0.0, 0.0, -1.0));
    {
        RayDesc_ naga_desc = desc;
        float naga_tmin = naga_desc.tmin;
        float naga_tmax = naga_desc.tmax;
        float3 naga_origin = naga_desc.origin;
        float3 naga_dir = naga_desc.dir;
        uint naga_flags = naga_desc.flags;
        bool naga_tmin_valid = (naga_tmin >= 0.0) && (naga_tmin <= naga_tmax) && !(((asuint(naga_tmin) & 2139095040) == 2139095040) && ((asuint(naga_tmin) & 0x7fffff) != 0));
        bool naga_tmax_valid = !(((asuint(naga_tmax) & 2139095040) == 2139095040) && ((asuint(naga_tmax) & 0x7fffff) != 0));
        bool naga_origin_valid = !any((((asuint(naga_origin) & 2139095040) == 2139095040) && ((asuint(naga_origin) & 0x7fffff) != 0)));
        bool naga_dir_valid = !any((((asuint(naga_dir) & 2139095040) == 2139095040) && ((asuint(naga_dir) & 0x7fffff) != 0)));
        bool naga_contains_opaque = ((naga_flags & 1) == 1);
        bool naga_contains_no_opaque = ((naga_flags & 2) == 2);
        bool naga_contains_cull_opaque = ((naga_flags & 64) == 64);
        bool naga_contains_cull_no_opaque = ((naga_flags & 128) == 128);
        bool naga_contains_cull_front = ((naga_flags & 32) == 32);
        bool naga_contains_cull_back = ((naga_flags & 16) == 16);
        bool naga_contains_skip_triangles = ((naga_flags & 256) == 256);
        bool naga_contains_skip_aabbs = ((naga_flags & 512) == 512);
        bool naga_contains_skip_triangles_aabbs =  (naga_contains_skip_aabbs && naga_contains_skip_triangles) ;
        bool naga_contains_skip_triangles_cull =  (naga_contains_cull_front && naga_contains_skip_triangles) || (naga_contains_cull_front && naga_contains_cull_back) || (naga_contains_cull_back && naga_contains_skip_triangles) ;
        bool naga_contains_multiple_opaque =  (naga_contains_cull_no_opaque && naga_contains_opaque) || (naga_contains_cull_no_opaque && naga_contains_no_opaque) || (naga_contains_cull_no_opaque && naga_contains_cull_opaque) || (naga_contains_cull_opaque && naga_contains_opaque) || (naga_contains_cull_opaque && naga_contains_no_opaque) || (naga_contains_no_opaque && naga_contains_opaque) ;
        if (naga_tmin_valid && naga_tmax_valid && naga_origin_valid && naga_dir_valid && !(naga_contains_skip_triangles_aabbs || naga_contains_skip_triangles_cull || naga_contains_multiple_opaque)) {
            naga_query_init_tracker_for_rq = 1;
            rq.TraceRayInline(first, naga_desc.flags, naga_desc.cull_mask, RayDescFromRayDesc_(naga_desc));
        }
    }
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        bool _e15 = false;
        {
            bool naga_has_initialized = ((naga_query_init_tracker_for_rq & 1) == 1);
            bool naga_has_finished = ((naga_query_init_tracker_for_rq & 4) == 4);
            if (naga_has_initialized && !naga_has_finished) {
                _e15 = rq.Proceed();
                naga_query_init_tracker_for_rq = naga_query_init_tracker_for_rq | 2;
                if (!_e15) { naga_query_init_tracker_for_rq = naga_query_init_tracker_for_rq | 4; }
        }}
        if (_e15) {
        } else {
            break;
        }
        {
        }
    }
    output.Store(0, asuint(GetCommittedIntersection(rq, naga_query_init_tracker_for_rq).t));
    {
        RayDesc_ naga_desc = desc;
        float naga_tmin = naga_desc.tmin;
        float naga_tmax = naga_desc.tmax;
        float3 naga_origin = naga_desc.origin;
        float3 naga_dir = naga_desc.dir;
        uint naga_flags = naga_desc.flags;
        bool naga_tmin_valid = (naga_tmin >= 0.0) && (naga_tmin <= naga_tmax) && !(((asuint(naga_tmin) & 2139095040) == 2139095040) && ((asuint(naga_tmin) & 0x7fffff) != 0));
        bool naga_tmax_valid = !(((asuint(naga_tmax) & 2139095040) == 2139095040) && ((asuint(naga_tmax) & 0x7fffff) != 0));
        bool naga_origin_valid = !any((((asuint(naga_origin) & 2139095040) == 2139095040) && ((asuint(naga_origin) & 0x7fffff) != 0)));
        bool naga_dir_valid = !any((((asuint(naga_dir) & 2139095040) == 2139095040) && ((asuint(naga_dir) & 0x7fffff) != 0)));
        bool naga_contains_opaque = ((naga_flags & 1) == 1);
        bool naga_contains_no_opaque = ((naga_flags & 2) == 2);
        bool naga_contains_cull_opaque = ((naga_flags & 64) == 64);
        bool naga_contains_cull_no_opaque = ((naga_flags & 128) == 128);
        bool naga_contains_cull_front = ((naga_flags & 32) == 32);
        bool naga_contains_cull_back = ((naga_flags & 16) == 16);
        bool naga_contains_skip_triangles = ((naga_flags & 256) == 256);
        bool naga_contains_skip_aabbs = ((naga_flags & 512) == 512);
        bool naga_contains_skip_triangles_aabbs =  (naga_contains_skip_aabbs && naga_contains_skip_triangles) ;
        bool naga_contains_skip_triangles_cull =  (naga_contains_cull_front && naga_contains_skip_triangles) || (naga_contains_cull_front && naga_contains_cull_back) || (naga_contains_cull_back && naga_contains_skip_triangles) ;
        bool naga_contains_multiple_opaque =  (naga_contains_cull_no_opaque && naga_contains_opaque) || (naga_contains_cull_no_opaque && naga_contains_no_opaque) || (naga_contains_cull_no_opaque && naga_contains_cull_opaque) || (naga_contains_cull_opaque && naga_contains_opaque) || (naga_contains_cull_opaque && naga_contains_no_opaque) || (naga_contains_no_opaque && naga_contains_opaque) ;
        if (naga_tmin_valid && naga_tmax_valid && naga_origin_valid && naga_dir_valid && !(naga_contains_skip_triangles_aabbs || naga_contains_skip_triangles_cull || naga_contains_multiple_opaque)) {
            naga_query_init_tracker_for_rq = 1;
            rq.TraceRayInline(second, naga_desc.flags, naga_desc.cull_mask, RayDescFromRayDesc_(naga_desc));
        }
    }
    uint2 loop_bound_1 = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound_1 == uint2(0u, 0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        bool _e21 = false;
        {
            bool naga_has_initialized = ((naga_query_init_tracker_for_rq & 1) == 1);
            bool naga_has_finished = ((naga_query_init_tracker_for_rq & 4) == 4);
            if (naga_has_initialized && !naga_has_finished) {
                _e21 = rq.Proceed();
                naga_query_init_tracker_for_rq = naga_query_init_tracker_for_rq | 2;
                if (!_e21) { naga_query_init_tracker_for_rq = naga_query_init_tracker_for_rq | 4; }
        }}
        if (_e21) {
        } else {
            break;
        }
        {
        }
    }
    output.Store(4, asuint(GetCommittedIntersection(rq, naga_query_init_tracker_for_rq).t));
    return;
}
