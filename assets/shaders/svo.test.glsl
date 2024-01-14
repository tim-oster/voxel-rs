#shader_type compute
#version 450

layout (local_size_x = 1) in;

layout (std430, binding = 11) readonly buffer buffer_in {
    float in_max_dst;
    bool in_cast_translucent;
    vec3 in_pos;
    vec3 in_dir;
};

struct Result {
    float t;
    uint value;
    int face_id;
    vec3 pos;
    vec2 uv;
    vec4 color;
    bool inside_voxel;
};

struct StackFrame {
    float t_min;
    uint ptr;
    int idx;
    uint parent_octant_idx;
    int scale;
    int is_child;
    int is_leaf;
};

layout (std430, binding = 12) buffer buffer_out {
    Result out_result;
    int out_stack_ptr;
    StackFrame out_stack[];
};

uniform sampler2DArray u_texture;

// override debug function to capture raytracing frames
#define OCTREE_RAYTRACE_DEBUG_FN(t_min, ptr, idx, parent_octant_idx, scale, is_child, is_leaf) \
    add_dbg_frame(t_min, ptr, idx, parent_octant_idx, scale, is_child, is_leaf)

// Creates a new frame on the stack by incrementing the stack pointer and storing all values in it.
void add_dbg_frame(float t_min, uint ptr, int idx, uint parent_octant_idx, int scale, bool is_child, bool is_leaf) {
    out_stack_ptr += 1;
    out_stack[out_stack_ptr].t_min = t_min;
    out_stack[out_stack_ptr].ptr = ptr;
    out_stack[out_stack_ptr].idx = idx;
    out_stack[out_stack_ptr].parent_octant_idx = parent_octant_idx;
    out_stack[out_stack_ptr].scale = scale;
    out_stack[out_stack_ptr].is_child = int(is_child);
    out_stack[out_stack_ptr].is_leaf = int(is_leaf);
}

#include "svo.glsl"

void main() {
    out_stack_ptr = -1;

    OctreeResult res;
    intersect_octree(in_pos, in_dir, in_max_dst, in_cast_translucent, u_texture, res);

    out_result.t = res.t;
    out_result.value = res.value;
    out_result.face_id = res.face_id;
    out_result.pos = res.pos;
    out_result.uv = res.uv;
    out_result.color = res.color;
    out_result.inside_voxel = res.inside_voxel;
}
