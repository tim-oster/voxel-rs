#version 460

in vec2 v_uv;

layout (location = 0) out vec4 color;

uniform mat4 u_view;
uniform float u_fovy;
uniform float u_aspect;

// lighting
uniform float u_ambient;
uniform vec3 u_light_dir;
uniform vec3 u_cam_pos;

const float specular_strength = 0.5;

// https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Buffer_backed
// https://www.khronos.org/opengl/wiki/Shader_Storage_Buffer_Object
layout (std430, binding = 0) buffer root_node {
    vec3 position;
    int descriptors[];
};

// source: https://www.iquilezles.org/www/articles/boxfunctions/boxfunctions.htm
vec2 intersect_box(in vec3 ro, in vec3 rd, in vec3 rad, out vec3 oN) {
    vec3 m = 1.0/rd;
    vec3 n = m*ro;
    vec3 k = abs(m)*rad;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

    float tN = max(max(t1.x, t1.y), t1.z);
    float tF = min(min(t2.x, t2.y), t2.z);

    if (tN>tF || tF<0.0) return vec2(-1.0);// no intersection

    oN = -sign(rd)*step(t1.yzx, t1.xyz)*step(t1.zxy, t1.xyz);

    return vec2(tN, tF);
}

const vec3 octree_pos = vec3(0.0);
const float octree_scale = 4;
const uint octree_desc[] = uint[](
0x00010200,
0x00000202
);

const vec3 octant_debug_colors[] = vec3[](
vec3(0.2, 0.2, 0.2),
vec3(0.0, 1.0, 0.0),
vec3(0.0, 0.0, 1.0),
vec3(0.0, 1.0, 1.0),
vec3(1.0, 0.0, 0.0),
vec3(1.0, 1.0, 0.0),
vec3(1.0, 0.0, 1.0),
vec3(1.0, 1.0, 1.0)
);

const float EPS = 0.00001;

float intersect_plane(vec3 ro, vec3 rd, vec3 po, vec3 pn) {
    float denom = dot(rd, pn);
    if (abs(denom) < EPS) return -1.0;
    float t = dot(po - ro, pn) / denom;
    if (t < 0.0) return -1.0;
    return t;
}

float intersect_octree(in vec3 ro, in vec3 rd, out vec3 oN, int ptr, float scale, vec3 pos) {
    // intersect against root octree first to find octant
    vec3 root_pos = pos - vec3(scale * 0.5);
    ro += root_pos;

    vec2 hit_2d = intersect_box(ro, rd, vec3(scale * 0.5), oN);
    float hit = hit_2d.x;
    if (hit < 0.0) return hit;

    vec3 hit_pos = ro + rd * hit;
    vec3 idx_3d = floor(clamp((hit_pos - root_pos) / scale * 2.0, EPS, 2.0-EPS));
    uint idx = uint(idx_3d.x) | (uint(idx_3d.y) << 1) | (uint(1-idx_3d.z) << 2);

    uint octant_bit = 1u << (idx);
    bool is_child = (octree_desc[ptr] & (octant_bit << 8)) != 0;
    bool is_leaf = (octree_desc[ptr] & octant_bit) != 0;

    if (is_child) {
        if (is_leaf) {
            oN = vec3(1.0);
            return hit;
        }

//        oN = vec3(octant_debug_colors[idx]);
        oN = vec3(0.5);
        int offset = int(octree_desc[ptr] & 0xffu);
        // TODO recursion not allowed
//        return intersect_octree(ro, rd, oN, ptr + offset, scale * 0.5, pos + idx_3d * scale * 0.5);
    } else {
        // TODO advance to sibling

        float x0 = intersect_plane(ro, rd, root_pos + scale * 0.5, vec3(1.0, 0.0, 0.0)) + EPS;
        if (x0 < hit_2d.x) x0 = 3e38;
        float y0 = intersect_plane(ro, rd, root_pos + scale * 0.5, vec3(0.0, 1.0, 0.0)) + EPS;
        if (y0 < hit_2d.x) y0 = 3e38;
        float z0 = intersect_plane(ro, rd, root_pos + scale * 0.5, vec3(0.0, 0.0, 1.0)) + EPS;
        if (z0 < hit_2d.x) z0 = 3e38;

        float hit = min(x0, min(z0, y0));
        if (hit > hit_2d.y) {
            // TODO exit the cube
            oN = vec3(0.0);
            return hit_2d.x;
        }

        vec3 hit_pos = ro + rd * hit;
        vec3 dst = (hit_pos - root_pos) / scale;

        vec3 idx_3d = floor(clamp(dst * 2.0, EPS, 2.0-EPS));
        uint idx = uint(idx_3d.x) | (uint(idx_3d.y) << 1) | (uint(1-idx_3d.z) << 2);

        uint octant_bit = 1u << (idx);
        bool is_child = (octree_desc[ptr] & (octant_bit << 8)) != 0;
        bool is_leaf = (octree_desc[ptr] & octant_bit) != 0;

        oN = vec3(octant_debug_colors[idx]);
    }

    return hit;
}

void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    uv.x *= u_aspect;
    uv *= tan(u_fovy * 0.5);

    vec3 ro = vec3(0.0, 0.0, 0.0);
    vec3 look_at = vec3(uv, -1.0);
    ro = (u_view * vec4(ro, 1.0)).xyz;
    look_at = (u_view * vec4(look_at, 1.0)).xyz;
    vec3 rd = normalize(look_at - ro);

    vec3 normal;
    float d = intersect_octree(ro, rd, normal, 0, octree_scale, octree_pos);

    color = vec4(normal, 1.0);

    //    vec3 frag_pos = ro + rd * d.x;
    //
    //    float diffuse = max(dot(normal, -u_light_dir), 0.0);
    //
    //    vec3 view_dir = normalize(frag_pos - u_cam_pos);
    //    vec3 reflect_dir = reflect(-u_light_dir, normal);
    //    float specular = pow(max(dot(view_dir, reflect_dir), 0.0), 256) * specular_strength;
    //
    //    float light = u_ambient + diffuse + specular;
    //
    //    color = vec4(vec3(step(0.0, d) * light), 1.0);
}
