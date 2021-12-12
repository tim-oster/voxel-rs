#define DEBUG 0

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

const int MAX_STEPS = 1000;

struct octree_result {
    float t;
    vec3 color;
    vec3 normal;
    vec3 pos;
    vec2 uv;
};

// https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Buffer_backed
// https://www.khronos.org/opengl/wiki/Shader_Storage_Buffer_Object
layout (std430, binding = 3) readonly buffer root_node {
    int descriptors[];
};

// TODO https://diglib.eg.org/bitstream/handle/10.2312/EGGH.EGGH89.061-073/061-073.pdf?sequence=1
// ideas from: https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/laine2010tr1_paper.pdf
void intersect_octree(in vec3 ro, in vec3 rd, float octree_scale, out octree_result res) {
    ro *= octree_scale;

    res.t = -1;
    res.color = vec3(0);

    // TODO try to what happens when using a positive coordinate system
    #if DEBUG
    vec2 minmax = intersect_box(ro, rd, octree_pos, vec3(0.5));
    if (minmax.x < 0.0 && minmax.y < 0.0) return;
    res.color = vec3(1, 0, 0);
    #endif

    // shift input coordinate system so that the octree spans from [1;2]
    ro += 1;

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
    int octant_mask = 0;
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
        int child_mask = (descriptors[ptr] >> 8) & 0xff;
        bool is_child = (child_mask & bit) != 0;
        bool is_leaf = (descriptors[ptr] & bit) != 0;

        if (is_child && t_min <= t_max) {
            if (is_leaf) {
                // TODO put after loop?

                int offset = int(uint(descriptors[ptr]) >> 17);
                int leaf_mask = descriptors[ptr] & 0xff;
                if ((descriptors[ptr] & 0x10000) != 0) {
                    offset = descriptors[ptr + offset];
                }
                ptr += offset;
                ptr += octant_idx - findLSB(leaf_mask);

                res.t = t_min;

                // TODO can be optimized?
                float tx_corner = (pos.x + scale_exp2) * tx_coef - tx_bias;
                float ty_corner = (pos.y + scale_exp2) * ty_coef - ty_bias;
                float tz_corner = (pos.z + scale_exp2) * tz_coef - tz_bias;
                float tc_min = max(max(tx_corner, ty_corner), tz_corner);

                if ((octant_mask & 1) != 0) pos.x = 3.0 - scale_exp2 - pos.x;
                if ((octant_mask & 2) != 0) pos.y = 3.0 - scale_exp2 - pos.y;
                if ((octant_mask & 4) != 0) pos.z = 3.0 - scale_exp2 - pos.z;

                // TODO can be optimized?
                if (tc_min == tx_corner) {
                    res.normal = vec3(1, 0, 0) * -sign(rd.x);
                    res.uv = vec2(
                    ((ro.z + rd.z * tx_corner) - pos.z) / scale_exp2,
                    ((ro.y + rd.y * tx_corner) - pos.y) / scale_exp2
                    );
                    if (res.normal.x > 0) res.uv.x = 1 - res.uv.x;
                } else if (tc_min == ty_corner) {
                    res.normal = vec3(0, 1, 0) * -sign(rd.y);
                    res.uv = vec2(
                    ((ro.x + rd.x * ty_corner) - pos.x) / scale_exp2,
                    ((ro.z + rd.z * ty_corner) - pos.z) / scale_exp2
                    );
                    if (res.normal.y < 0) res.uv.x = 1 - res.uv.x;
                } else {
                    res.normal = vec3(0, 0, 1) * -sign(rd.z);
                    res.uv = vec2(
                    ((ro.x + rd.x * tz_corner) - pos.x) / scale_exp2,
                    ((ro.y + rd.y * tz_corner) - pos.y) / scale_exp2
                    );
                    if (res.normal.z < 0) res.uv.x = 1 - res.uv.x;
                }

                res.pos.x = min(max(ro.x + t_min * rd.x, pos.x + epsilon), pos.x + scale_exp2 - epsilon);
                res.pos.y = min(max(ro.y + t_min * rd.y, pos.y + epsilon), pos.y + scale_exp2 - epsilon);
                res.pos.z = min(max(ro.z + t_min * rd.z, pos.z + epsilon), pos.z + scale_exp2 - epsilon);
                // undo initial coordinate system shift
                res.pos -= 1;
                res.pos /= octree_scale;

                res.color = vec3(
                float(descriptors[ptr] & 0xff) / 255.0,
                float((descriptors[ptr] >> 8) & 0xff) / 255.0,
                float((descriptors[ptr] >> 16) & 0xff) / 255.0
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

                // TODO convert everything to uint?
                int offset = int(uint(descriptors[ptr]) >> 17);
                if ((descriptors[ptr] & 0x10000) != 0) {
                    offset = descriptors[ptr + offset];
                }
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
