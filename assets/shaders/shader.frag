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
vec2 intersect_box(in vec3 ro, in vec3 rd, in vec3 pos, in vec3 rad) {
    ro -= pos + rad;

    vec3 m = 1.0/rd;
    vec3 n = m*ro;
    vec3 k = abs(m)*rad;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

    float tN = max(max(t1.x, t1.y), t1.z);
    float tF = min(min(t2.x, t2.y), t2.z);

    if (tN>tF || tF<0.0) return vec2(-1.0);// no intersection

    //    oN = -sign(rd)*step(t1.yzx, t1.xyz)*step(t1.zxy, t1.xyz);

    return vec2(tN, tF);
}

const vec3 octree_pos = vec3(0.0, 0.0, 0.0);
const float octree_scale = 1;
const uint octree_desc[] = uint[](
0x00011600,
0x00011600,
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

const float EPS = 1e-5;
const float MAX_FLOAT = 3e38;
const int MAX_STEPS = 100;

float intersect_plane(vec3 ro, vec3 rd, vec3 po, vec3 pn) {
    float denom = dot(rd, pn);
    if (abs(denom) < EPS) return -1.0;
    float t = dot(po - ro, pn) / denom;
    if (t < 0.0) return -1.0;
    return t;
}

uint get_octant_idx(vec3 hit_pos, vec3 octree_pos, float scale) {
    vec3 to = hit_pos - octree_pos;
    vec3 in_octree = floor(clamp(to / scale * 2.0, EPS, 2.0-EPS));
    uint idx = uint(in_octree.x) | (uint(in_octree.y) << 1) | (uint(1-in_octree.z) << 2);// TODO do 1-x on z?
    return idx;
}

vec3 get_offset_from_octant_idx(uint idx) {
    return vec3(idx & 0x1u, (idx >> 1) & 0x1u, 1.0 - float((idx >> 2) & 0x1u));
}

void get_octant_flags(uint desc, uint idx, out bool is_child, out bool is_leaf) {
    uint bit = 1u << idx;
    is_child = (desc & (bit << 8)) != 0;
    is_leaf = (desc & bit) != 0;
}

vec3 intersect_octree_v1(in vec3 ro, in vec3 rd) {
    // TODO does is work for negative coordinates?
    // TODO support tracing from inside the octree

    // TODO figure out proper stack size
    const int STACK_SIZE = 100;
    int[STACK_SIZE] ptr_stack;
    float[STACK_SIZE] t_max_stack;// TODO can be removed by raycasting again in POP?
    vec3[STACK_SIZE] pos_stack;// TODO can be removed because it is a relative calculation?
    int stack_ptr = 0;

    int ptr = 0;
    float scale = octree_scale;
    float half_scale = scale * 0.5;

    vec3 pos = octree_pos;
    vec2 minmax = intersect_box(ro, rd, pos, vec3(half_scale));
    if (minmax.x < 0.0) return vec3(0);

    float t = minmax.x;
    float t_max = minmax.y;

    for (int i = 0; i < MAX_STEPS; ++i) {
        vec3 hit_pos = ro + rd * t;
        uint idx = get_octant_idx(hit_pos, pos, scale);
        bool is_child, is_leaf;
        get_octant_flags(octree_desc[ptr], idx, is_child, is_leaf);

        if (is_child) {
            // If the hit child is a leaf node, stop the ray tracing and return t.
            if (is_leaf) {
                return vec3(1);
            }

            // PUSH: if it is not a leaf node, enter the child octree and continue ray tracing.
            ptr_stack[stack_ptr] = ptr;
            t_max_stack[stack_ptr] = t_max;
            pos_stack[stack_ptr] = pos;
            ++stack_ptr;

            int ptr_incr = int((octree_desc[ptr] & 0xffff0000u) >> 16);
            ptr += ptr_incr;

            vec3 offset = get_offset_from_octant_idx(idx) * half_scale;
            pos += offset;

            // TODO simplify this?
            scale *= 0.5;
            half_scale *= 0.5;

            vec2 minmax = intersect_box(ro, rd, pos, vec3(half_scale));
            t = minmax.x + EPS;
            t_max = minmax.y;
        } else {
            // ADVANCE: if the no child is found in the current octant, skip to the next octant along the ray
            // while staying in the same parent octree

            // Intersect against every "middle-plane" of the cube. Sort results in ascending order to find the
            // next intersecting octant along the ray. Discard all results that are below the initial hit against
            // the octree as they are outside the encapsulating cube.
            float x0 = intersect_plane(ro, rd, pos + half_scale, vec3(1.0, 0.0, 0.0)) + EPS;
            if (x0 < t + EPS) x0 = MAX_FLOAT;
            float y0 = intersect_plane(ro, rd, pos + half_scale, vec3(0.0, 1.0, 0.0)) + EPS;
            if (y0 < t + EPS) y0 = MAX_FLOAT;
            float z0 = intersect_plane(ro, rd, pos + half_scale, vec3(0.0, 0.0, 1.0)) + EPS;
            if (z0 < t + EPS) z0 = MAX_FLOAT;

            t = min(t_max + EPS, min(x0, min(y0, z0)));

            // POP: move up one layer into the parent octree and restore the old state from the stack.
            while (t > t_max && stack_ptr > 0) {
                --stack_ptr;
                ptr = ptr_stack[stack_ptr];
                t_max = t_max_stack[stack_ptr];
                pos = pos_stack[stack_ptr];

                // TODO simplify this?
                scale *= 2.0;
                half_scale *= 2.0;
            }

            if (t > t_max - EPS) {
                return vec3(0);
            }
        }
    }

    return vec3(1, 0, 0);
}

// TODO https://diglib.eg.org/bitstream/handle/10.2312/EGGH.EGGH89.061-073/061-073.pdf?sequence=1
// ideas from: https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/laine2010tr1_paper.pdf
vec3 intersect_octree_v2(in vec3 ro, in vec3 rd) {
    // prevents divide by zero
    if (abs(rd.x) < EPS) rd.x = EPS * sign(rd.x);
    if (abs(rd.y) < EPS) rd.y = EPS * sign(rd.y);
    if (abs(rd.z) < EPS) rd.z = EPS * sign(rd.z);

    // abs to prevent negative directions from inversing the ordering of min(tx, ty, tz). Negative components
    // will reuslt in negative coefficients, which always leads to smaller t values, regardless of what the other
    // positive components are.
    float tx_coef = 1.0 / abs(rd.x);
    float ty_coef = 1.0 / abs(rd.y);
    float tz_coef = 1.0 / abs(rd.z);

    float tx_bias = tx_coef * ro.x;
    float ty_bias = ty_coef * ro.y;
    float tz_bias = tz_coef * ro.z;

    // TODO does is work for negative coordinates?
    // TODO support tracing from inside the octree

    // TODO figure out proper stack size
    const int STACK_SIZE = 100;
    int[STACK_SIZE] ptr_stack;
    float[STACK_SIZE] t_max_stack;// TODO can be removed by raycasting again in POP?
    vec3[STACK_SIZE] pos_stack;// TODO can be removed because it is a relative calculation?
    int stack_ptr = 0;

    int ptr = 0;
    float scale = octree_scale;
    float half_scale = scale * 0.5;

    // TODO only for debugging
    vec3 pos = octree_pos;
    vec2 minmax = intersect_box(ro, rd, pos, vec3(half_scale));
    if (minmax.x < 0.0) return vec3(0);
    // TODO only for debugging

    // Negative directions mirror the results, hence a flip mask is required to flip them back into
    // the actual, all postive directions, result. Biases are also flipped between a plane at t0 or t1
    // depending on the direction. This ensures that positive ray directions look at planes at 0 and negative
    // directions always look at planes at 1.
    int flip_mask = 0;
    if (rd.x < 0) flip_mask ^= 1, tx_bias = 1 * tx_coef - tx_bias;
    if (rd.y < 0) flip_mask ^= 2, ty_bias = 1 * ty_coef - ty_bias;
    if (rd.z < 0) flip_mask ^= 4, tz_bias = 1 * tz_coef - tz_bias;

    float t_min = max(max(-tx_bias, -ty_bias), -tz_bias);

    int idx = 0;
    if (t_min > 0.5 * tx_coef - tx_bias) idx ^= 1;
    if (t_min > 0.5 * ty_coef - ty_bias) idx ^= 2;
    if (t_min > 0.5 * tz_coef - tz_bias) idx ^= 4;

    return octant_debug_colors[idx ^ flip_mask];

    //    vec3 pos = octree_pos;
    //    vec2 minmax = intersect_box(ro, rd, pos, vec3(half_scale));
    //    if (minmax.x < 0.0) return -1.0;
    //
    //    float t = minmax.x;
    //    float t_max = minmax.y;
    //
    //    for (int i = 0; i < MAX_STEPS; ++i) {
    //        vec3 hit_pos = ro + rd * t;
    //        uint idx = get_octant_idx(hit_pos, pos, scale);
    //        bool is_child, is_leaf;
    //        get_octant_flags(octree_desc[ptr], idx, is_child, is_leaf);
    //
    //        if (is_child) {
    //            // If the hit child is a leaf node, stop the ray tracing and return t.
    //            if (is_leaf) {
    //                return t;
    //            }
    //
    //            // PUSH: if it is not a leaf node, enter the child octree and continue ray tracing.
    //            ptr_stack[stack_ptr] = ptr;
    //            t_max_stack[stack_ptr] = t_max;
    //            pos_stack[stack_ptr] = pos;
    //            ++stack_ptr;
    //
    //            int ptr_incr = int((octree_desc[ptr] & 0xffff0000u) >> 16);
    //            ptr += ptr_incr;
    //
    //            vec3 offset = get_offset_from_octant_idx(idx) * half_scale;
    //            pos += offset;
    //
    //            // TODO simplify this?
    //            scale *= 0.5;
    //            half_scale *= 0.5;
    //
    //            vec2 minmax = intersect_box(ro, rd, pos, vec3(half_scale));
    //            t = minmax.x + EPS;
    //            t_max = minmax.y;
    //        } else {
    //            // ADVANCE: if the no child is found in the current octant, skip to the next octant along the ray
    //            // while staying in the same parent octree
    //
    //            // Intersect against every "middle-plane" of the cube. Sort results in ascending order to find the
    //            // next intersecting octant along the ray. Discard all results that are below the initial hit against
    //            // the octree as they are outside the encapsulating cube.
    //            float x0 = intersect_plane(ro, rd, pos + half_scale, vec3(1.0, 0.0, 0.0)) + EPS;
    //            if (x0 < t + EPS) x0 = MAX_FLOAT;
    //            float y0 = intersect_plane(ro, rd, pos + half_scale, vec3(0.0, 1.0, 0.0)) + EPS;
    //            if (y0 < t + EPS) y0 = MAX_FLOAT;
    //            float z0 = intersect_plane(ro, rd, pos + half_scale, vec3(0.0, 0.0, 1.0)) + EPS;
    //            if (z0 < t + EPS) z0 = MAX_FLOAT;
    //
    //            t = min(t_max + EPS, min(x0, min(y0, z0)));
    //
    //            // POP: move up one layer into the parent octree and restore the old state from the stack.
    //            while (t > t_max && stack_ptr > 0) {
    //                --stack_ptr;
    //                ptr = ptr_stack[stack_ptr];
    //                t_max = t_max_stack[stack_ptr];
    //                pos = pos_stack[stack_ptr];
    //
    //                // TODO simplify this?
    //                scale *= 2.0;
    //                half_scale *= 2.0;
    //            }
    //
    //            if (t > t_max - EPS) {
    //                return -1.0; // TODO
    //            }
    //        }
    //    }

    return vec3(1, 0, 0);
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

    #if 0
    color = vec4(intersect_octree_v1(ro, rd), 1);
    #else
    color = vec4(intersect_octree_v2(ro, rd), 1);
    #endif

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
