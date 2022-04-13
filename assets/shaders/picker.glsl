#version 460

#include "svo.glsl"

layout (local_size_x = 1) in;

layout (std430, binding = 1) writeonly buffer picker_data {
    vec3 pos;
    int parent_index;
    int octant_idx;
};

uniform vec3 u_cam_pos;
uniform vec3 u_cam_dir;

void main() {
    octree_result res;
    intersect_octree(u_cam_pos, u_cam_dir, 30, res);

    if (res.t > 0) {
        pos = res.pos;
        parent_index = res.parent_index;
        octant_idx = res.octant_idx;
    } else {
        // max float = no hit
        pos = vec3(3.402823466E+38);
        parent_index = 0;
        octant_idx = 0;
    }
}