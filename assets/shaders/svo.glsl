// Pre-calculated normals per face in order [x-, x+, y-, y+, z-, z+].
/*const*/ vec3 FACE_NORMALS[6] = vec3[6](
vec3(-1, 0, 0),
vec3(1, 0, 0),
vec3(0, -1, 0),
vec3(0, 1, 0),
vec3(0, 0, -1),
vec3(0, 0, 1)
);

// Pre-calculated tangents per face in order [x-, x+, y-, y+, z-, z+].
/*const*/ vec3 FACE_TANGENTS[6] = vec3[6](
vec3(0, 0, 1),
vec3(0, 0, -1),
vec3(1, 0, 0),
vec3(1, 0, 0),
vec3(-1, 0, 0),
vec3(1, 0, 0)
);

// Pre-calculated bi-tangents per face in order [x-, x+, y-, y+, z-, z+].
/*const*/ vec3 FACE_BITANGENTS[6] = vec3[6](
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

#if !defined(OCTREE_RAYTRACE_DEBUG_FN)
// NOP implementation if the debug function has not been defined until here
#define OCTREE_RAYTRACE_DEBUG_FN(t_min, ptr, idx, parent_octant_idx, scale, is_child, is_leaf, crossed_boundary, next_ptr);
#endif

// Material contains rendering properties for which textures to load per side and what parameters to use for lighting.
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

#define SVO_TYPE_ESVO 1
#define SVO_TYPE_CSVO 2

#if SVO_TYPE == SVO_TYPE_ESVO
#include "svo.esvo.glsl"
#elif SVO_TYPE == SVO_TYPE_CSVO
#include "svo.csvo.glsl"
#else
#error SVO_TYPE not valid
#endif
