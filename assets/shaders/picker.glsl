#shader_type compute
#version 450

#include "svo.glsl"

layout (local_size_x = 1) in;

// subset of svo.glsl's OctreeResult
struct PickerResult {
    float dst;
    bool inside_voxel;
    vec3 pos;
    vec3 normal;
};
layout (std430, binding = 1) writeonly buffer picker_output {
    PickerResult results[100];
};

struct PickerTask {
    float max_dst;// stop raytracing if nothing was hit past this distance
    vec3 pos;// ray origin
    vec3 dir;// ray direction
};
layout (std430, binding = 3) readonly buffer picker_input {
    PickerTask tasks[100];
};

uniform sampler2DArray u_texture;

void main() {
    // pick one task per invocation group
    uint index = gl_GlobalInvocationID.x;
    PickerTask task = tasks[index];

    // cast into octree and stop at translucent blocks
    OctreeResult res;
    intersect_octree(task.pos, task.dir, task.max_dst, false, u_texture, res);

    // write hit into result buffer
    if (res.t > 0) {
        results[index].dst = res.t;
        results[index].inside_voxel = res.inside_voxel;
        results[index].pos = res.pos;
        results[index].normal = FACE_NORMALS[res.face_id];
    } else {
        results[index].dst = -1;
        results[index].inside_voxel = false;
        results[index].pos = vec3(0);
        results[index].normal = vec3(0);
    }
}
