// Pre-calculated normals per face in order [x-, x+, y-, y+, z-, z+].
const vec3 FACE_NORMALS[6] = vec3[6](
vec3(-1, 0, 0),
vec3(1, 0, 0),
vec3(0, -1, 0),
vec3(0, 1, 0),
vec3(0, 0, -1),
vec3(0, 0, 1)
);

// Pre-calculated tangents per face in order [x-, x+, y-, y+, z-, z+].
const vec3 FACE_TANGENTS[6] = vec3[6](
vec3(0, 0, 1),
vec3(0, 0, -1),
vec3(1, 0, 0),
vec3(1, 0, 0),
vec3(-1, 0, 0),
vec3(1, 0, 0)
);

// Pre-calculated bi-tangents per face in order [x-, x+, y-, y+, z-, z+].
const vec3 FACE_BITANGENTS[6] = vec3[6](
vec3(0, 1, 0),
vec3(0, 1, 0),
vec3(0, 0, 1),
vec3(0, 0, 1),
vec3(0, 1, 0),
vec3(0, 1, 0)
);

struct OctreeResult {
    float t;// distance along the ray in world space
    uint value;// hit octree value
    int face_id;// face index of the voxel that was hit (0=-x, 1=+x, 2=-y, 3=+y, 4=-z, 5=+z)
    vec3 pos;// hit position in world space (ro + t*rd)
    vec2 uv;// uv coordinate on the voxel face
    vec4 color;// texture color of the hit point
    float lod;// lod that was used for texture lookup
    bool inside_voxel;// true if ray is cast from within a voxel
};

// https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Buffer_backed
// https://www.khronos.org/opengl/wiki/Shader_Storage_Buffer_Object
layout (std430, binding = 0) readonly buffer RootNode {
    float octree_scale;// Size of one leaf node in an octree from [0;1]. Calculated by `2^(-octree_depth)`.
    uint descriptors[];// Serialized octree bytes. See `src/world/svo.rs` for details on the format.
};

// Material contains rendering properties for which textures to load per side and what paramteres to use for lighting.
struct Material {
    float specular_pow;
    float specular_strength;

    int tex_top;
    int tex_side;
    int tex_bottom;

    int tex_top_normal;
    int tex_side_normal;
    int tex_bottom_normal;
};

layout (std430, binding = 2) readonly buffer MaterialRegistry {
    Material materials[];
};

#if !defined(OCTREE_RAYTRACE_DEBUG_FN)
// NOP implementation if the debug function has not been defined until here
#define OCTREE_RAYTRACE_DEBUG_FN(t_min, ptr, idx, parent_octant_idx, scale, is_child, is_leaf);
#endif

// TODO explain IEEE 754 and the advantage of having numbers [1;2]
// TODO write documentation
// TODO explain PUSH, ADVANCE, POP
// TODO document parameters & return value (t=-1)
// TODO include reference to svo data format in rs file
// List of sources:
//  - Samuli Laine and Tero Karras. 2010 "Efficient sparse voxel octrees"
//      - https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/laine2010i3d_paper.pdf
//  - Samuli Laine and Tero Karras. 2010 "Efficient sparse voxel octrees – Analysis, Extensions, and Implementation"
//      - https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/laine2010tr1_paper.pdf
void intersect_octree(vec3 ro, vec3 rd, float max_dst, bool cast_translucent, sampler2DArray textures, out OctreeResult res) {
    const int MAX_STEPS = 1000;
    // The algorithm relies on the IEEE 754 floating point bit representation. Since it using single precision (f32),
    // there are only 23 bits in the fractional part.
    const int MAX_SCALE = 23;
    // The smallest ray increment can only be exp2(-22) because of the fractional bit size of single precision floats.
    // Hence an epsilon value of exp2(-23) can be used for floating point operations.
    const float epsilon = exp2(-MAX_SCALE);

    // rescale inputs to be [0;1]
    ro *= octree_scale;
    max_dst *= octree_scale;

    // initialise all return values
    res.t = -1;
    res.value = 0;
    res.face_id = 0;
    res.pos = vec3(0);
    res.uv = vec2(0);
    res.color = vec4(0);
    res.inside_voxel = false;

    // Shift input coordinate system so that the octree spans from [1;2]. Doing so allows the alogrithm to work directly
    // on the mantiassa/fractional bits of the float bits.
    ro += 1;

    // stacks to implement PUSH & POP for step into and out of the child octants
    uint[MAX_SCALE] ptr_stack;
    uint[MAX_SCALE] parent_octant_idx_stack;
    float[MAX_SCALE] t_max_stack;

    uint ptr = 0;// current pointer inside the SVO data structure
    uint parent_octant_idx = 0;// the child index [0;7] of the current octant's parent octant

    // Scale is the mantiassa bit of the current ray step size. Starts at bit 22, which with an exponent of 0, is
    // equal to 1.5, or in case of this algorithm is interpreted as 0.5. A more intuitive representation would be a
    // scale that increments from 0..22 as the algorithm descends into the octree, but choosing the inverted form allows
    // for optimizing the POP implementation below.
    int scale = MAX_SCALE - 1;
    float scale_exp2 = 0.5;// = exp2(scale - MAX_SCALE)

    // In case a leaf has a texture with transparency, the ray can pass through several leaf voxels. When that happens,
    // these variables keep track of how many adjecent leafs of the same type/value the ray has passed through. Every
    // identical leaf after the first one is skipped.
    uint last_leaf_value = -1;
    int adjecent_leaf_count = 0;

    // Prevent division by zero by making sure that rd is never less than epsilon, both in positive and negative
    // direction. Use bit-magic to copy the rd sign to the epsilon value.
    int sign_mask = 1 << 31;
    int epsilon_bits_without_sign = floatBitsToInt(epsilon) & ~sign_mask;
    if (abs(rd.x) < epsilon) rd.x = intBitsToFloat(epsilon_bits_without_sign | (floatBitsToInt(rd.x) & sign_mask));
    if (abs(rd.y) < epsilon) rd.y = intBitsToFloat(epsilon_bits_without_sign | (floatBitsToInt(rd.y) & sign_mask));
    if (abs(rd.z) < epsilon) rd.z = intBitsToFloat(epsilon_bits_without_sign | (floatBitsToInt(rd.z) & sign_mask));

    // To calculate octant intersections, the algorithm needs to know the distance `t` along every axis until the next
    // octant at the current step size is reached. This can be expressed as `rx(t) = rx + t * dx` for the x-axis.
    // This equation however is solving for the target position. In this algorithm, the target position is known as
    // `pos.x + scale_exp2`. So instead of solving for x, it must solve for t to determine how far away the target
    // position is from the current position on a given axis: `tx(x) = (x - rx) / dx`.
    //
    // To optimize calculations, the result can be rewritten as `tx(x) = x * (1/dx) - (rx/dx)` allowing for
    // pre-calculating the coefficient of x and the bias once. This means that calculating the next interception
    // distnance is one FMA-operation (fused multiply-add) per axis: `x * tx_coef - tx_bias`.
    //
    // Because t always increases, the bias only has to be caclculated once with the ray origin position.
    //
    // Ensure that ray directions are always negative, this is required so that the mirroring logic below works.
    float tx_coef = 1.0 / -abs(rd.x);
    float ty_coef = 1.0 / -abs(rd.y);
    float tz_coef = 1.0 / -abs(rd.z);
    float tx_bias = tx_coef * ro.x;
    float ty_bias = ty_coef * ro.y;
    float tz_bias = tz_coef * ro.z;

    // To keep the implementation logic simple, all positive ray directions are mirrored. This allows for stepping
    // through the octree without having to keep the sign of the direction into account. Whenever information about a
    // voxel is interpreted, the calculated octant_mask is used to undo the mirroring.
    //
    // To mirror correctly, the equestion from above needs to be altered as well. Given that the octree positions are
    // within [1;2], `tx(x) = x * tx_coef - tx_bias` can be rewritten as `tx'(x) = (3 - x) * tx_coef - tx_bias`.
    // This can be simplified to only adjust the bias `tx'(x) = x * tx_coef - (3 * tx_coef - tx_bias)`. Hence,
    // to mirror the equation for one axis, the bias has to be rewritten as `tx_bias = x * tx_coef - tx_bias`.
    //
    // Using the negative direction has the advantegous property that the octant position (see `pos` below) defines
    // the upper bound of the current octant that the algorithm casts against. Since this bound is calculated more often
    // than the lower bound (only required when it is already determined that a leaf octant was hit), using negative
    // directions saves the add operations (see calculation of tc_min in leaf hit logic).
    int octant_mask = 0;
    if (rd.x > 0) octant_mask ^= 1, tx_bias = 3.0 * tx_coef - tx_bias;
    if (rd.y > 0) octant_mask ^= 2, ty_bias = 3.0 * ty_coef - ty_bias;
    if (rd.z > 0) octant_mask ^= 4, tz_bias = 3.0 * tz_coef - tz_bias;

    // Calculate distance t_min from ro at which the ray enters the octant. t_min will be increased while casting the
    // ray and represents the final dst result. It might be negative when ro is within the octree. Since rd is always
    // negative in the distance equation, the first interception is at the axis aligned planes at (2, 2, 2).
    float t_min = max(max(2.0 * tx_coef - tx_bias, 2.0 * ty_coef - ty_bias), 2.0 * tz_coef - tz_bias);
    t_min = max(0, t_min);
    // Calculate the distance t_max at which the ray leaves the octree. Since rd is always negative in the distance
    // equation, the exit interception is at the axis aligned planes at (1, 1, 1).
    float t_max = min(min(tx_coef - tx_bias, ty_coef - ty_bias), tz_coef - tz_bias);

    // idx is the current octant index inside its parent. Every octant can have 8 children. The index is calculated by
    // allocating one bit per axis and setting it to 0 or 1 depending on the position of the octant on the given axis
    // (e.g. (0,0,0) => 0b000 => 0, (1,0,1) => 0b101 => 5, (0,1,0) => 0b010 => 4).
    int idx = 0;
    // pos keeps track of the current octant's position inside the octree. Each component is within [1;2].
    vec3 pos = vec3(1.0);
    // Since the ray casts through the octree in the inverted direction from 2 -> 1, the intersection and index
    // calculation logic needs to be inverted as well. If t_min is less than the distance to the center of every
    // axis (1.5), then globally the entry position is in the second half of the octant. To indicate this, the idx flag
    // is updated. Additionally, pos needs to be set to 1.5 to reflect that fact.
    if (t_min < 1.5 * tx_coef - tx_bias) idx ^= 1, pos.x = 1.5;
    if (t_min < 1.5 * ty_coef - ty_bias) idx ^= 2, pos.y = 1.5;
    if (t_min < 1.5 * tz_coef - tz_bias) idx ^= 4, pos.z = 1.5;

    // Start stepping through the octree until a voxel is hit or max steps are reached.
    // Also interrupt if scale exceeds its limit, i.e. the octree is too deep.
    for (int i = 0; i < MAX_STEPS && scale < MAX_SCALE; ++i) {
        if (max_dst >= 0 && t_min > max_dst) {
            // early return if max_dst is set and reached
            return;
        }

        // Because the ray direction is inverted, pos defines the corner of the cube where the ray will exit.
        float tx_corner = pos.x * tx_coef - tx_bias;
        float ty_corner = pos.y * ty_coef - ty_bias;
        float tz_corner = pos.z * tz_coef - tz_bias;
        // The smallest distance across axes determines the exit distance of the current octant.
        float tc_max = min(min(tx_corner, ty_corner), tz_corner);

        // get octant index and its bit index
        uint octant_idx = idx ^ octant_mask;
        uint bit = 1u << octant_idx;

        // lookup the descriptor for the current octant
        uint descriptor = descriptors[ptr + (parent_octant_idx / 2)];
        if ((parent_octant_idx % 2) != 0) {
            descriptor >>= 16;
        }
        descriptor &= 0xffffu;

        bool is_child = (descriptor & (bit << 8)) != 0;
        bool is_leaf = (descriptor & bit) != 0;

        OCTREE_RAYTRACE_DEBUG_FN(t_min/octree_scale, ptr, idx, parent_octant_idx, scale, is_child, is_leaf);

        // check if a child octant was hit
        if (is_child && t_min <= t_max) {
            // flag inside_voxel if the octree starts at a leaf with no steps along the ray
            if (is_leaf && t_min == 0) {
                res.inside_voxel = true;
            }

            // if the child is a leaf, calculate the result
            if (is_leaf && t_min > 0) {
                // fetch pointer for leaf value
                uint next_ptr = descriptors[ptr + 4 + parent_octant_idx];
                if ((next_ptr & (1u << 31u)) != 0) {
                    // use as relative offset if relative bit is set
                    next_ptr = ptr + 4 + parent_octant_idx + (next_ptr & 0x7fffffffu);
                }
                next_ptr = next_ptr + 4 + octant_idx;

                // fetch leaf value
                uint value = descriptors[next_ptr];

                // Because the ray direction is inverted, use pos and the current octant scale to calculate the entry
                // distance of the hit leaf.
                float tx_corner = (pos.x + scale_exp2) * tx_coef - tx_bias;
                float ty_corner = (pos.y + scale_exp2) * ty_coef - ty_bias;
                float tz_corner = (pos.z + scale_exp2) * tz_coef - tz_bias;
                // The smallest distance across axes determines the entry distance of the hit leaf.
                float tc_min = max(max(tx_corner, ty_corner), tz_corner);

                // Use octant_mask to undo mirroring of pos.
                vec3 pos = pos;
                if ((octant_mask & 1) != 0) pos.x = 3.0 - scale_exp2 - pos.x;
                if ((octant_mask & 2) != 0) pos.y = 3.0 - scale_exp2 - pos.y;
                if ((octant_mask & 4) != 0) pos.z = 3.0 - scale_exp2 - pos.z;

                // Calculate the face_id & uv coords by comparing tc_min against every entry corner, to figure out
                // which face the ray hit. UVs are the distance between the leaf corner position and the ray hit
                // position rescaled by the current scale.
                int face_id;
                vec2 uv;
                if (tc_min == tx_corner) {
                    face_id = (floatBitsToInt(rd.x) >> 31) & 1;
                    uv = vec2((ro.z + rd.z * tx_corner) - pos.z, (ro.y + rd.y * tx_corner) - pos.y) / scale_exp2;
                    if (rd.x > 0) uv.x = 1 - uv.x;
                } else if (tc_min == ty_corner) {
                    face_id = 2 | ((floatBitsToInt(rd.y) >> 31) & 1);
                    uv = vec2((ro.x + rd.x * ty_corner) - pos.x, (ro.z + rd.z * ty_corner) - pos.z) / scale_exp2;
                    if (rd.y > 0) uv.y = 1 - uv.y;
                } else {
                    face_id = 4 | ((floatBitsToInt(rd.z) >> 31) & 1);
                    uv = vec2((ro.x + rd.x * tz_corner) - pos.x, (ro.y + rd.y * tz_corner) - pos.y) / scale_exp2;
                    if (rd.z < 0) uv.x = 1 - uv.x;
                }

                // Look up the material to determine which texture to use for the face that was hit.
                Material mat = materials[value];
                int tex_id = mat.tex_side;
                if (face_id == 3) { tex_id = mat.tex_top; }
                else if (face_id == 2) { tex_id = mat.tex_bottom; }

                // rescale t_min ([0;1]) to world scale
                float dst = t_min / octree_scale;
                // calculate custom texture lod interpolation factor
                float tex_lod = smoothstep(15, 25, dst) * (dst-15) * 0.05;

                #if SHADER_COMPILE_TYPE != SHADER_TYPE_COMPUTE
                // use deriviate of t because uv is not continuous (resets after every voxel)
                vec4 tex_color_a = textureGrad(textures, vec3(uv, float(tex_id)), vec2(dFdx(dst), 0), vec2(dFdy(dst), 0));
                vec4 tex_color_b = textureLod(textures, vec3(uv, float(tex_id)), tex_lod);
                vec4 tex_color = mix(tex_color_b, tex_color_b, smoothstep(15, 25, dst));
                #else
                vec4 tex_color = textureLod(textures, vec3(uv, float(tex_id)), tex_lod);
                #endif

                // If texel is not translucent, or cast_translucent = false, calculate the result and stop the
                // algorithm. Ignore the leaf if it is not the first of its kind, when casting translucent voxels.
                bool first_of_kind = adjecent_leaf_count == 0 || value != last_leaf_value;
                if ((tex_color.a > 0 || !cast_translucent) && first_of_kind) {
                    res.t = dst;
                    res.face_id = face_id;
                    res.uv = uv;
                    res.value = value;
                    res.color = tex_color;
                    res.lod = tex_lod;

                    // Clamp final `ro + t_min * rd` between octant start & end position to mitigature floating point
                    // errors.
                    res.pos.x = min(max(ro.x + t_min * rd.x, pos.x + epsilon), pos.x + scale_exp2 - epsilon);
                    res.pos.y = min(max(ro.y + t_min * rd.y, pos.y + epsilon), pos.y + scale_exp2 - epsilon);
                    res.pos.z = min(max(ro.z + t_min * rd.z, pos.z + epsilon), pos.z + scale_exp2 - epsilon);

                    // undo initial coordinate system shift & rescale
                    res.pos -= 1;
                    res.pos /= octree_scale;

                    return;
                }

                // If the texel is translucent and cast_translucent=true, keep track of adjacent leaves.
                ++adjecent_leaf_count;
                last_leaf_value = value;
            } else {
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
                    parent_octant_idx_stack[scale] = parent_octant_idx;
                    t_max_stack[scale] = t_max;

                    uint next_ptr = descriptors[ptr + 4 + parent_octant_idx];
                    if ((next_ptr & (1u << 31)) != 0) {
                        // use as relative offset if relative bit is set
                        next_ptr = ptr + 4 + parent_octant_idx + (next_ptr & 0x7fffffffu);
                    }
                    ptr = next_ptr;

                    --scale;
                    parent_octant_idx = octant_idx;
                    scale_exp2 = half_scale;

                    idx = 0;
                    if (t_min < tx_center) idx ^= 1, pos.x += scale_exp2;
                    if (t_min < ty_center) idx ^= 2, pos.y += scale_exp2;
                    if (t_min < tz_center) idx ^= 4, pos.z += scale_exp2;

                    t_max = tv_max;
                    continue;
                }
            }
        } else {
            // if no leaf is found, reset the adjcent leaf counter
            adjecent_leaf_count = 0;
            last_leaf_value = -1;
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
            if ((step_mask & 1) != 0) differing_bits |= floatBitsToUint(pos.x) ^ floatBitsToUint(pos.x + scale_exp2);
            if ((step_mask & 2) != 0) differing_bits |= floatBitsToUint(pos.y) ^ floatBitsToUint(pos.y + scale_exp2);
            if ((step_mask & 4) != 0) differing_bits |= floatBitsToUint(pos.z) ^ floatBitsToUint(pos.z + scale_exp2);
            scale = (floatBitsToInt(differing_bits) >> 23) - 127;// ingores sign bit because always 0/positive
            scale_exp2 = intBitsToFloat((scale - MAX_SCALE + 127) << 23);

            ptr = ptr_stack[scale];
            parent_octant_idx = parent_octant_idx_stack[scale];
            t_max = t_max_stack[scale];

            int shx = floatBitsToInt(pos.x) >> scale;
            int shy = floatBitsToInt(pos.y) >> scale;
            int shz = floatBitsToInt(pos.z) >> scale;
            pos.x = intBitsToFloat(shx << scale);
            pos.y = intBitsToFloat(shy << scale);
            pos.z = intBitsToFloat(shz << scale);
            idx = (shx & 1) | ((shy & 1) << 1) | ((shz & 1) << 2);

            // TODO set h?
        }
    }
}
