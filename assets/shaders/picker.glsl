#version 460

#include "svo.glsl"

layout (local_size_x = 1) in;

struct PickerResult {
    float dst;
    vec3 pos;
    vec3 normal;
};
layout (std430, binding = 1) writeonly buffer picker_output {
    PickerResult results[50];
};

struct PickerTask {
    float max_dst;
    vec3 pos;
    vec3 dir;
};
layout (std430, binding = 3) readonly buffer picker_input {
    PickerTask tasks[50];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    PickerTask task = tasks[index];

    octree_result res;
    intersect_octree(task.pos, task.dir, task.max_dst, false, res);

    if (res.t > 0) {
        results[index].dst = res.t;
        results[index].pos = res.pos;
        results[index].normal = FACE_NORMALS[res.face_id];
    } else {
        results[index].dst = -1;
        results[index].pos = vec3(0);
        results[index].normal = vec3(0);
    }
}
