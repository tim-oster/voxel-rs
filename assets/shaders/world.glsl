#shader_type vertex
#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;

out vec2 v_uv;

void main() {
    v_uv = uv;
    gl_Position = vec4(position, 1.0);
}

// ------------------------------------------------------------

#shader_type fragment
#version 460

#include "svo.glsl"

in vec2 v_uv;

layout (location = 0) out vec4 color;

uniform mat4 u_view;
uniform float u_fovy;
uniform float u_aspect;

// lighting
uniform float u_ambient;
uniform vec3 u_light_dir;
uniform vec3 u_cam_pos;

// block highlighting
uniform vec3 u_highlight_pos;

vec4 trace_ray(vec3 ro, vec3 rd) {
    octree_result res;
    intersect_octree(ro, rd, -1, true, res);

    if (res.t < 0) {
        return vec4(0);
    }
    if (floor(res.pos) == floor(u_highlight_pos)) {
        const float thickness = 1./16.;
        vec2 local = abs(res.uv - 0.5) * 2;
        float lmax = max(local.x, local.y);
        if (lmax > 1.0 - thickness) {
            return vec4(1-color.rgb, 1);
        }
    }

    Material mat = materials[res.value];
    int tex_normal_id = mat.tex_side_normal;
    if (res.face_id == 3) { tex_normal_id = mat.tex_top_normal; }
    else if (res.face_id == 2) { tex_normal_id = mat.tex_bottom_normal; }

    vec3 normal = FACE_NORMALS[res.face_id];
    vec3 tangent = FACE_TANGENTS[res.face_id];
    vec3 bitangent = FACE_BITANGENTS[res.face_id];
    if (tex_normal_id != -1) {
        vec3 tex = texture(u_texture, vec3(res.uv, float(tex_normal_id))).xzy;// blue = up -> y axis
        // NOTE: since automatic mipmap generation is used, the resulting textures do not look that nicely. To
        // prevent this from being noticable, texture lookups at the base mip level are forced for close distances.
        if (res.t < 20) tex = textureLod(u_texture, vec3(res.uv, float(tex_normal_id)), smoothstep(15, 20, res.t)).xzy;
        tex = normalize(tex * 2 - 1);
        normal = tex.x * tangent + tex.y * normal + tex.z * bitangent;
    }

    float diffuse = max(dot(normal, -u_light_dir), 0.0);

    vec3 view_dir = normalize(res.pos - u_cam_pos);
    vec3 reflect_dir = reflect(-u_light_dir, normal);
    float specular = pow(max(dot(view_dir, reflect_dir), 0.0), mat.specular_pow) * mat.specular_strength;

    float shadow = 1;
    if (res.t < 100/octree_scale) {
        octree_result shadow_res;
        intersect_octree(res.pos + normal*0.001, -u_light_dir, -1, true, shadow_res);
        shadow = shadow_res.t < 0 ? 1.0 : 0.0;
    }

    float light = clamp(u_ambient + (diffuse + specular) * shadow, 0.0, 1.0);
    res.color.rgb *= light;
    return res.color;
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

    color = trace_ray(ro, rd);
}
