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
    float t;// distance along the ray in world space (no hit if t == -1)
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

// Resolve the pointer to the given octant from the current ptr. Actual format is explained in `src/world/svo.rs`.
/*inline*/ uint get_octant_ptr(uint ptr, uint idx) {
    uint next_ptr = descriptors[ptr + 4 + idx];
    if ((next_ptr & (1u << 31)) != 0) {
        // use as relative offset if relative bit is set
        next_ptr = ptr + 4 + idx + (next_ptr & 0x7fffffffu);
    }
    return next_ptr;
}

// Parses the descriptor for the given parent octant and looks up child & leaf mask for octant_idx.
// Actual format is explained in `src/world/svo.rs`.
/*inline*/ bvec2 parse_descriptor(uint ptr, uint parent_octant_idx, uint octant_idx) {
    uint bit = 1u << octant_idx;

    // lookup the descriptor for the current octant
    uint descriptor = descriptors[ptr + (parent_octant_idx / 2)];
    if ((parent_octant_idx % 2) != 0) {
        descriptor >>= 16;
    }
    descriptor &= 0xffffu;

    bool is_child = (descriptor & (bit << 8)) != 0;
    bool is_leaf = (descriptor & bit) != 0;

    return bvec2(is_child, is_leaf);
}

// Intersects the given ray (defined by ro & rd) against the octree in SVO format. It uses a modified implementation of
// the raytracer described in Laine and Karras "Efficient sparse voxel octrees". In contrast to their implementation,
// contour and level of detail were removed in favor of a dynamically loadable, and more memory efficient data
// structure that allows for real-time voxel manipulation.
//
// Params:
//  ro                  ray origin
//  rd                  ray direction (normalized)
//  max_dst             if > 0, limits the distance the ray travels
//  cast_translucent    if true, rays can pass trough translucent texels on a voxel
//  textures            texture array to lookup material textures from
//  out res             returns ray hit result - no hit occured when res.t == -1
//
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

    // Shift input coordinate system so that the octree spans from [1;2). Doing so allows the alogrithm to work directly
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
    // within [1;2), `tx(x) = x * tx_coef - tx_bias` can be rewritten as `tx'(x) = (3 - x) * tx_coef - tx_bias`.
    // This can be simplified to only adjust the bias `tx'(x) = x * tx_coef - (3 * tx_coef - tx_bias)`. Hence,
    // to mirror the equation for one axis, the bias has to be rewritten as `tx_bias = x * tx_coef - tx_bias`.
    //
    // Using negative directions has the advantegous property that the pos vector decreases over time. This allows the
    // POP phase implementation to round down floats using bit operations, which is more efficient than alternative
    // implementations. The negative property also simplifies the calculation of corner points, which happens
    // frequently.
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
    // pos keeps track of the current octant's position inside the octree. Each component is within [1;2).
    vec3 pos = vec3(1.0);
    // Since the ray casts through the octree in the inverted direction from 2 -> 1, the intersection and index
    // calculation logic needs to be inverted as well. If t_min is less than the distance to the center of every
    // axis (1.5), then globally the entry position is in the second half of the octant. To indicate this, the idx flag
    // is updated. Additionally, pos needs to be set to 1.5 to reflect that fact.
    if (t_min < 1.5 * tx_coef - tx_bias) idx ^= 1, pos.x = 1.5;
    if (t_min < 1.5 * ty_coef - ty_bias) idx ^= 2, pos.y = 1.5;
    if (t_min < 1.5 * tz_coef - tz_bias) idx ^= 4, pos.z = 1.5;

    // Start stepping through the octree until a voxel is hit or max steps are reached.
    for (int i = 0; i < MAX_STEPS; ++i) {
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
        bvec2 desc = parse_descriptor(ptr, parent_octant_idx, octant_idx);
        bool is_child = desc.x;
        bool is_leaf = desc.y;

        OCTREE_RAYTRACE_DEBUG_FN(t_min/octree_scale, ptr, idx, parent_octant_idx, scale, is_child, is_leaf);

        // check if a child octant was hit
        if (is_child && t_min <= t_max) {
            // flag inside_voxel if the octree starts at a leaf with no steps along the ray
            if (is_leaf && t_min == 0) {
                res.inside_voxel = true;
            }

            // if the child is a leaf, calculate the result
            if (is_leaf && t_min > 0) {
                // phase: HIT
                // calculate leaf intersection data and return

                // fetch pointer for leaf value
                uint next_ptr = get_octant_ptr(ptr, parent_octant_idx);
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
                // intersect with octant and descend into it

                // Use corner coordinates (upper bound of the current octant) and add half the scale to them to get
                // the center position.
                float half_scale = scale_exp2 * 0.5;
                float tx_center = half_scale * tx_coef + tx_corner;
                float ty_center = half_scale * ty_coef + ty_corner;
                float tz_center = half_scale * tz_coef + tz_corner;

                // do not exceed current parent octant
                float tv_max = min(t_max, tc_max);

                if (t_min <= tv_max) {
                    // phase: PUSH

                    // "push" current values onto the stack
                    ptr_stack[scale] = ptr;
                    parent_octant_idx_stack[scale] = parent_octant_idx;
                    t_max_stack[scale] = t_max;

                    // fetch next octant pointer
                    ptr = get_octant_ptr(ptr, parent_octant_idx);

                    // descend into next scale & update values
                    --scale;
                    parent_octant_idx = octant_idx;
                    scale_exp2 = half_scale;

                    // Use center values to determine the next idx & pos using the same logic as done during the setup
                    // phase of the algorithm.
                    idx = 0;
                    if (t_min < tx_center) idx ^= 1, pos.x += scale_exp2;
                    if (t_min < ty_center) idx ^= 2, pos.y += scale_exp2;
                    if (t_min < tz_center) idx ^= 4, pos.z += scale_exp2;

                    // udpate t_max to not allow exceeding the child octant that was descended into
                    t_max = tv_max;

                    // make sure that ADVANCE phase is skipped
                    continue;
                }
            }
        } else {
            // if no leaf is found, reset the adjcent leaf counter
            adjecent_leaf_count = 0;
            last_leaf_value = -1;
        }

        // phase: ADVANCE
        // if nothing was hit, advance the ray to the next sibling

        // Figure out which corner is the closest and build the step_mask using it. Also adjust the pos (upper bound)
        // to inverse direction to reflect the next upper bound correctly.
        int step_mask = 0;
        if (tc_max >= tx_corner) step_mask ^= 1, pos.x -= scale_exp2;
        if (tc_max >= ty_corner) step_mask ^= 2, pos.y -= scale_exp2;
        if (tc_max >= tz_corner) step_mask ^= 4, pos.z -= scale_exp2;

        // advance t_min and perform step on idx
        t_min = tc_max;
        idx ^= step_mask;

        // If the next idx does not align with the step mask, i.e. the step caused the ray to leave the parent octant,
        // pop the current stack and ascend to the highest parent that the ray exists and proceed with the next sibling.
        if ((idx & step_mask) != 0) {
            // phase: POP

            // Numbers in [1;2) have the advantage that they can be directly encoded in the mantiassa in IEEE 754.
            // The mantissa is defined by `(2^0)+(2^-1)+(2^-2)+...+(2^-n)`, where n=23 for single precision. So if the
            // 23rd bit is set, the number is 1.5; if the 23rd and 22nd bits are set, it is 1.75; and so on. This is
            // convenient because each bit can hence represent one layer of the octree.
            //
            // Because ADVANCE stepped outside the parent octant, adding scale_exp2 to pos will cause a higher bit of
            // the mantiassa to be flipped (e.g. `1.25 + 0.25 = 1.5 <=> (2^0)+(2^-2) + (2^-2) = (2^0)+(2^-1)`). Using
            // this approach, a differing_bits mask can be calculated by comparing the current position against the new
            // one on every axis.
            uint differing_bits = 0;
            if ((step_mask & 1) != 0) differing_bits |= floatBitsToUint(pos.x) ^ floatBitsToUint(pos.x + scale_exp2);
            if ((step_mask & 2) != 0) differing_bits |= floatBitsToUint(pos.y) ^ floatBitsToUint(pos.y + scale_exp2);
            if ((step_mask & 4) != 0) differing_bits |= floatBitsToUint(pos.z) ^ floatBitsToUint(pos.z + scale_exp2);

            // Assuming pos and pos+scale_exp2 are withing [1;2), differing_bits can be used to extract the new scale
            // by finding the most significant bit. The advantage of this approach is, that it can ascend multiple
            // parents at once instead of having to waste cycles tracking back through the stack one by one.
            //
            // Example: given a position = 1.375 (0b011) and scale_exp2 = 0.125 (= scale 21 = 2^(21-23))
            // Adding scale_exp2 to position yields 1.5 (0b100) and differing_bits = 0b111. The most significant bit of
            // differing_bits (in IEEE 754 single precision) is 23. Hence, the algorithm can continue at scale = 23
            // directly without having to ascend through scale = 22 first.
            scale = findMSB(differing_bits);
            scale_exp2 = exp2(scale - MAX_SCALE);

            // scale can only exceed its limit, if the most significant bit is outside of the float's mantissa, in which
            // case the position is >= 2. Thus the ray left the octree without hitting anything.
            if (scale >= MAX_SCALE) {
                return;
            }

            // "pop" values from the stack at the new scale
            ptr = ptr_stack[scale];
            parent_octant_idx = parent_octant_idx_stack[scale];
            t_max = t_max_stack[scale];

            // Floor all positions to given scale to truncate previous information from when the ray was within a child
            // octant. Because the position becomes smaller as the ray advances, this efficient round down mechansim
            // works. For example: 1.375 (0b011) rounded to scale=22 becomes 1.25 (0b010).
            int shx = floatBitsToInt(pos.x) >> scale;
            int shy = floatBitsToInt(pos.y) >> scale;
            int shz = floatBitsToInt(pos.z) >> scale;
            pos.x = intBitsToFloat(shx << scale);
            pos.y = intBitsToFloat(shy << scale);
            pos.z = intBitsToFloat(shz << scale);

            // Recalcuate the new, ascended to, parent's index by checking if the bit, that was rounded to, is set.
            // For example: if rounding 1.75 (0b11) to 1.5 (0b10) for scale=23, then the 23rd bit is set. However,
            // rounding 1.25 (0b01) to 1.0 (0b00) for the same scale, does not thave the 23rd bit set. Hence, they are
            // contribute to the new idx respectively.
            idx = (shx & 1) | ((shy & 1) << 1) | ((shz & 1) << 2);
        }
    }
}
