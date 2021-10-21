#version 460

#define DEBUG 1

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

#if DEBUG
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
#endif

const vec3 octree_pos = vec3(0.0, 0.0, 0.0);
const float octree_scale = 1;

// pointer offset      children  leaves
// 0000_0000 0000_0000 0000_0000 0000_0000
const int octree_desc[] = int[](
0x00010500,
0x00030101,
0x0,
0x00020101,
0x00ff66,
0x996666
);

const vec3 octant_debug_colors[] = vec3[](
vec3(0.2, 0.2, 0.2),
vec3(1.0, 0.0, 0.0),
vec3(0.0, 1.0, 0.0),
vec3(1.0, 1.0, 0.0),
vec3(0.0, 0.0, 1.0),
vec3(1.0, 0.0, 1.0),
vec3(0.0, 1.0, 1.0),
vec3(1.0, 1.0, 1.0)
);

const int MAX_STEPS = 100;

struct octree_result {
    float t;
    vec3 color;
    vec3 normal;
};

// TODO https://diglib.eg.org/bitstream/handle/10.2312/EGGH.EGGH89.061-073/061-073.pdf?sequence=1
// ideas from: https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/laine2010tr1_paper.pdf
void intersect_octree(in vec3 ro, in vec3 rd, out octree_result res) {
    res.t = -1;
    res.color = vec3(0);

    // TODO try to what happens when using a positive coordinate system
    #if DEBUG
    vec2 minmax = intersect_box(ro, rd, octree_pos, vec3(0.5));
    if (minmax.x < 0.0 && minmax.y < 0.0) return;
    res.color = vec3(1, 0, 0);
    #endif
    ro += vec3(1);

    const int MAX_STACK_DEPTH = 23;
    int[MAX_STACK_DEPTH] ptr_stack;
    float[MAX_STACK_DEPTH] t_max_stack;
    int stack_ptr = 0;

    const float epsilon = exp2(-MAX_STACK_DEPTH);

    int ptr = 0;
    int scale = MAX_STACK_DEPTH - 1;
    float scale_exp2 = 0.5;

    // prevents divide by zero
    if (abs(rd.x) < epsilon) rd.x = epsilon * sign(rd.x);
    if (abs(rd.y) < epsilon) rd.y = epsilon * sign(rd.y);
    if (abs(rd.z) < epsilon) rd.z = epsilon * sign(rd.z);

    // abs to prevent negative directions from inversing the ordering of min(tx, ty, tz). Negative components
    // will reuslt in negative coefficients, which always leads to smaller t values, regardless of what the other
    // positive components are.
    float tx_coef = 1.0 / -abs(rd.x);
    float ty_coef = 1.0 / -abs(rd.y);
    float tz_coef = 1.0 / -abs(rd.z);

    float tx_bias = tx_coef * ro.x;
    float ty_bias = ty_coef * ro.y;
    float tz_bias = tz_coef * ro.z;

    // Negative directions mirror the results, hence a flip mask is required to flip them back into
    // the actual, all postive directions, result. Biases are also flipped between a plane at t0 or t1
    // depending on the direction. This ensures that positive ray directions look at planes at 0 and negative
    // directions always look at planes at 1.
    int octant_mask = 7;
    if (rd.x > 0) octant_mask ^= 1, tx_bias = 3.0 * tx_coef - tx_bias;
    if (rd.y > 0) octant_mask ^= 2, ty_bias = 3.0 * ty_coef - ty_bias;
    if (rd.z > 0) octant_mask ^= 4, tz_bias = 3.0 * tz_coef - tz_bias;

    float t_min = max(max(2.0 * tx_coef - tx_bias, 2.0 * ty_coef - ty_bias), 2.0 * tz_coef - tz_bias);
    float t_max = min(min(tx_coef - tx_bias, ty_coef - ty_bias), tz_coef - tz_bias);
    t_min = max(0, t_min);
    //t_max = min(1, t_max); // TODO why?

    int idx = 0;
    vec3 pos = vec3(1.0);
    if (t_min < 1.5 * tx_coef - tx_bias) idx ^= 1, pos.x = 1.5;
    if (t_min < 1.5 * ty_coef - ty_bias) idx ^= 2, pos.y = 1.5;
    if (t_min < 1.5 * tz_coef - tz_bias) idx ^= 4, pos.z = 1.5;

    for (int i = 0; i < MAX_STEPS && scale < MAX_STACK_DEPTH; ++i) {
        float tx_corner = pos.x * tx_coef - tx_bias;
        float ty_corner = pos.y * ty_coef - ty_bias;
        float tz_corner = pos.z * tz_coef - tz_bias;
        float tc_max = min(min(tx_corner, ty_corner), tz_corner);

        int octant_idx = idx ^ octant_mask;
        int bit = 1 << octant_idx;
        int child_mask = (octree_desc[ptr] >> 8) & 0xff;
        bool is_child = (child_mask & bit) != 0;
        bool is_leaf = (octree_desc[ptr] & bit) != 0;

        if (is_child && t_min <= t_max) {
            int offset = (octree_desc[ptr] & 0xffff0000) >> 16;

            // TODO intersection at back-face?!
            if (is_leaf) {
                int leaf_mask = octree_desc[ptr] & 0xff;
                ptr += offset;
                ptr += octant_idx - findLSB(leaf_mask);

                res.t = t_min;

                // TODO can be optimized?
                if (tc_max == tx_corner) {
                    res.normal = vec3(1, 0, 0) * -sign(rd.x);
                } else if (tc_max == ty_corner) {
                    res.normal = vec3(0, 1, 0) * -sign(rd.y);
                } else {
                    res.normal = vec3(0, 0, 1) * -sign(rd.z);
                }

                res.color = vec3(
                float((octree_desc[ptr] >> 16) & 0xff) / 255.0,
                float((octree_desc[ptr] >> 8) & 0xff) / 255.0,
                float(octree_desc[ptr] & 0xff) / 255.0
                );

                return;
            }

            // INTERSECT
            float tv_max = min(t_max, tc_max);
            float half_scale = scale_exp2 * 0.5;
            float tx_center = half_scale * tx_coef + tx_corner;
            float ty_center = half_scale * ty_coef + ty_corner;
            float tz_center = half_scale * tz_coef + tz_corner;

            if (t_min <= tv_max) {
                // PUSH

                // TODO this is guarded in the original
                ptr_stack[scale] = ptr;
                t_max_stack[scale] = t_max;

                ptr += offset;
                ptr += octant_idx - findLSB(child_mask);

                idx = 0;
                --scale;
                scale_exp2 = half_scale;

                if (t_min < tx_center) idx ^= 1, pos.x += scale_exp2;
                if (t_min < ty_center) idx ^= 2, pos.y += scale_exp2;
                if (t_min < tz_center) idx ^= 4, pos.z += scale_exp2;

                t_max = tv_max;
                continue;
            }
        }

        // ADVANCE
        int step_mask = 0;
        if (tc_max >= tx_corner) step_mask ^= 1, pos.x -= scale_exp2;
        if (tc_max >= ty_corner) step_mask ^= 2, pos.y -= scale_exp2;
        if (tc_max >= tz_corner) step_mask ^= 4, pos.z -= scale_exp2;

        t_min = tc_max;
        idx ^= step_mask;

        if ((idx & step_mask) != 0) {
            // POP

            uint differing_bits = 0;
            if ((step_mask & 1) != 0) differing_bits |= floatBitsToInt(pos.x) ^ floatBitsToInt(pos.x + scale_exp2);
            if ((step_mask & 2) != 0) differing_bits |= floatBitsToInt(pos.y) ^ floatBitsToInt(pos.y + scale_exp2);
            if ((step_mask & 4) != 0) differing_bits |= floatBitsToInt(pos.z) ^ floatBitsToInt(pos.z + scale_exp2);
            scale = (floatBitsToInt(differing_bits) >> 23) - 127;
            scale_exp2 = intBitsToFloat((scale - MAX_STACK_DEPTH + 127) << 23);

            ptr = ptr_stack[scale];
            t_max = t_max_stack[scale];

            int shx = floatBitsToInt(pos.x) >> scale;
            int shy = floatBitsToInt(pos.y) >> scale;
            int shz = floatBitsToInt(pos.z) >> scale;
            pos.x = intBitsToFloat(shx << scale);
            pos.y = intBitsToFloat(shy << scale);
            pos.z = intBitsToFloat(shz << scale);
            idx = (shx & 1) | ((shy & 1) << 1) | ((shz & 1) << 2);
        }
    }
}

void main() {
    // TODO next steps:
    //  - fix backface intersections
    //	- add normal & position information
    //  - add uv calculation
    //	- add basic lighting
    //	- scale to world
    //	- generate model from magica voxel
    //	- add sources & doc/explainations

    // TODO test with https://sketchfab.com/3d-models/summer-hamlet-voxel-diorama-758293a06ecc4a9787107d554a497b06

    vec2 uv = v_uv * 2.0 - 1.0;
    uv.x *= u_aspect;
    uv *= tan(u_fovy * 0.5);

    vec3 ro = vec3(0.0, 0.0, 0.0);
    vec3 look_at = vec3(uv, -1.0);
    ro = (u_view * vec4(ro, 1.0)).xyz;
    look_at = (u_view * vec4(look_at, 1.0)).xyz;
    vec3 rd = normalize(look_at - ro);

    octree_result res;
    intersect_octree(ro, rd, res);

    float diffuse = max(dot(res.normal, -u_light_dir), 0.0);
    float light = u_ambient + diffuse;
    color = vec4(res.color, 1) * light;

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
