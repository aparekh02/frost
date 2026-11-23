// Frost Bridge hardware render pipeline (the "gravity" graphics plane).
//
// Unlike the compute kernels in kernels.metal, this path uses the GPU's fixed-function
// rasterizer: a vertex shader + a fragment shader, run by an MTLRenderCommandEncoder that
// draws into a texture *aliasing the capsule's arena framebuffer*. The frame therefore
// lands in unified memory the capsule can read with no copy — this is the accelerated
// graphics path a 2D game actually targets.
//
// Geometry is "vertex-pulled" from an arena buffer indexed by vertex_id, so no vertex
// descriptor / stride negotiation is needed. One GVertex is 48 bytes; see the matching
// `struct GVertex` in crates/workloads/src/bin/gravity.rs.
#include <metal_stdlib>
using namespace metal;

struct GVertex {
    float4 color;    // straight-alpha RGBA
    float2 pos;      // pixel-space position of this vertex
    float2 center;   // disc center in pixels (radius > 0), else ignored
    float  radius;   // > 0: shade+clip as a disc (sprite); <= 0: flat quad (bg / floor)
    float  _pad[3];  // pad to 48 B / 16-B alignment
};

struct GUniforms { float2 viewport; }; // framebuffer size in pixels

struct VOut {
    float4 pos [[position]];
    float4 color;
    float2 world;   // interpolated pixel position
    float2 center;
    float  radius;
};

vertex VOut g_vert(uint vid [[vertex_id]],
                   device const GVertex* v [[buffer(0)]],
                   constant GUniforms& u   [[buffer(1)]]) {
    GVertex g = v[vid];
    // pixel space -> clip space (y down -> y up)
    float2 ndc = float2(g.pos.x / u.viewport.x * 2.0 - 1.0,
                        1.0 - g.pos.y / u.viewport.y * 2.0);
    VOut o;
    o.pos = float4(ndc, 0.0, 1.0);
    o.color = g.color;
    o.world = g.pos;
    o.center = g.center;
    o.radius = g.radius;
    return o;
}

fragment float4 g_frag(VOut in [[stage_in]]) {
    if (in.radius <= 0.0) {
        return in.color;                 // flat geometry (background gradient, floor)
    }
    // Analytic anti-aliased disc: a sprite rendered by the fragment shader.
    float d = distance(in.world, in.center);
    float aa = max(fwidth(d), 1.0);
    float cov = clamp((in.radius - d) / aa + 0.5, 0.0, 1.0);
    if (cov <= 0.0) discard_fragment();
    float2 n = (in.center - in.world) / max(in.radius, 1.0);
    float lit = clamp(dot(normalize(float2(-0.5, -0.7)), n) * 0.5 + 0.6, 0.0, 1.0);
    float3 c = in.color.rgb * (0.35 + 0.65 * lit);
    return float4(c, in.color.a * cov);
}
