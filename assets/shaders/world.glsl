#shader_type compute
#version 450

#include "svo.glsl"

#define PI 3.141592
#define HALF_PI 1.570796

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
layout (rgba32f, binding = 0) uniform image2D render_target;

uniform mat4 u_view;// converts world to view space
uniform float u_fovy;
uniform float u_aspect;// screen width / height
uniform sampler2DArray u_texture;

// lighting
uniform float u_ambient;// ambient light intensity - to fake global illumination
uniform vec3 u_light_dir;// sun light direction
uniform vec3 u_cam_pos;// world space position of the camera
uniform bool u_render_shadows;// enables secondary ray casting
uniform float u_shadow_distance;// distance until which shadows are rendered

// block highlighting
uniform vec3 u_highlight_pos;// world space position of the block the player is highlighting

vec4 trace_ray(vec3 ro, vec3 rd, out bool hit) {
    OctreeResult res;
    intersect_octree(ro, rd, -1, true, u_texture, res);

    hit = res.t != -1;

    if (res.t < 0) {
        // return early on no hit
        return vec4(0);
    }
    if (floor(res.pos) == floor(u_highlight_pos)) {
        // if block is highlighted, draw a white outline around it
        const float thickness = 1./16.;
        vec2 local = abs(res.uv - 0.5) * 2;
        float lmax = max(local.x, local.y);
        if (lmax > 1.0 - thickness) {
            return vec4(1);
        }
    }

    // select the normal texture for the given face of the voxel's material
    Material mat = materials[res.value];
    int tex_normal_id = mat.tex_side_normal;
    if (res.face_id == 3) { tex_normal_id = mat.tex_top_normal; }
    else if (res.face_id == 2) { tex_normal_id = mat.tex_bottom_normal; }

    // select precalculated vectors for the face
    vec3 normal = FACE_NORMALS[res.face_id];
    vec3 tangent = FACE_TANGENTS[res.face_id];
    vec3 bitangent = FACE_BITANGENTS[res.face_id];

    // if a normal texture is set, use its value as the normal vector
    if (tex_normal_id != -1) {
        vec3 tex = textureLod(u_texture, vec3(res.uv, float(tex_normal_id)), res.lod).xzy;

        // map [0;1] to [-1;1]
        tex = normalize(tex * 2 - 1);

        // blue = up -> y axis
        normal = tex.x * tangent + tex.y * normal + tex.z * bitangent;
    }

    // calculate diffuse lighting using the sun light direction vector
    float diffuse = max(dot(normal, -u_light_dir), 0.0);

    // calculate specular lighting using camera position and sun light direction
    vec3 view_dir = normalize(res.pos - u_cam_pos);
    vec3 reflect_dir = reflect(-u_light_dir, normal);
    float specular = pow(max(dot(view_dir, reflect_dir), 0.0), mat.specular_pow) * mat.specular_strength;

    // Calculate shadow by casting another ray from the previous hit location towards the sun. Skip if the hit is too
    // far away.
    float shadow = 1;
    if (u_render_shadows && res.t < u_shadow_distance) {
        OctreeResult shadow_res;
        intersect_octree(res.pos + normal*0.001, -u_light_dir, -1, true, u_texture, shadow_res);
        shadow = shadow_res.t < 0 ? 1.0 : 0.0;
    }

    // combine light calculations and color
    float light = clamp(u_ambient + (diffuse + specular) * shadow, 0.0, 1.0);
    res.color.rgb *= light;
    return res.color;
}

vec3 get_sky_color(vec3 rd) {
    const vec3 SKY_COLOR = vec3(135.0/255.0, 206.0/255.0, 235.0/255.0);// hex: #87CEEB
    const vec3 HORIZON_COLOR = mix(vec3(1), SKY_COLOR, 0.3);

    // get angle between xz plane and look dir
    vec3 p = normalize(vec3(rd.x, 0, rd.z));
    float a = acos(dot(rd, p) / (abs(length(rd))) * abs(length(p)));

    // calculate linear gradient based on angle
    float grad = a / HALF_PI;

    // use easing function to skew white part towards horizon
    grad = 1 - pow(1 - grad, 3);

    // interpolate between horizon and sky color
    return mix(HORIZON_COLOR, SKY_COLOR, grad);
}

void main() {
    // convert uv from [0;1] to [-1;1], apply screen aspect ratio & vertical FoV
    vec2 uv = vec2(gl_GlobalInvocationID.xy) / imageSize(render_target);
    uv = uv * 2.0 - 1.0;
    uv.x *= u_aspect;
    uv *= tan(u_fovy * 0.5);

    vec3 ro = vec3(0.0, 0.0, 0.0);
    vec3 look_at = vec3(uv, -1.0);

    // convert ray origin to view space
    vec4 ro_view = u_view * vec4(ro, 1.0);
    ro = ro_view.xyz / ro_view.w;

    // convert look_at to view space
    vec4 look_at_view = u_view * vec4(look_at, 1.0);
    look_at = look_at_view.xyz / look_at_view.w;

    // cast ray from origin to look_at
    vec3 rd = normalize(look_at - ro);
    bool hit = false;

    vec4 color = trace_ray(ro, rd, hit);

    // calculate sky color if nothing was hit
    if (!hit) {
        vec3 sky = get_sky_color(rd);
        color = vec4(sky, 1.0);
    }

    imageStore(render_target, ivec2(gl_GlobalInvocationID.xy), color);
}
