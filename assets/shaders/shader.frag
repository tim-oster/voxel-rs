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
uniform sampler2DArray u_texture;

// block highlighting
uniform vec3 u_highlight_pos;

struct Material {
    float specular_pow;
    float specular_strength;
    uint tex_top;
    uint tex_side;
    uint tex_bottom;
};

layout (std430, binding = 2) readonly buffer material_registry {
    Material materials[];
};

void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    uv.x *= u_aspect;
    uv *= tan(u_fovy * 0.5);

    vec3 ro = vec3(0.0, 0.0, 0.0);
    vec3 look_at = vec3(uv, -1.0);
    ro = (u_view * vec4(ro, 1.0)).xyz;
    look_at = (u_view * vec4(look_at, 1.0)).xyz;
    vec3 rd = normalize(look_at - ro);

    octree_result res;
    intersect_octree(ro, rd, -1, res);

    if (res.t < 0) {
        color = vec4(0);
        return;
    }

    Material mat = materials[res.value];

    uint tex_id = mat.tex_side;
    int face_id = int(dot(abs(res.normal.yxz), vec3(0, 2, 4)));
    if (dot(res.normal, vec3(1)) < 0) face_id += 1;
    if (face_id == 0) tex_id = mat.tex_top;
    else if (face_id == 1) tex_id = mat.tex_bottom;
    vec4 tex_color = texture(u_texture, vec3(res.uv, float(tex_id)));

    float diffuse = max(dot(res.normal, -u_light_dir), 0.0);

    vec3 view_dir = normalize(res.pos - u_cam_pos);
    vec3 reflect_dir = reflect(-u_light_dir, res.normal);
    float specular = pow(max(dot(view_dir, reflect_dir), 0.0), mat.specular_pow) * mat.specular_strength;

    octree_result shadow_res;
    intersect_octree(res.pos + res.normal*0.0005, -u_light_dir, -1, shadow_res);
    float shadow = shadow_res.t < 0 ? 1.0 : 0.0;

    float light = u_ambient + (diffuse + specular) * shadow;
    color = tex_color * light;

    if (floor(res.pos) == floor(u_highlight_pos)) {
        const float thickness = 1./16.;
        vec2 local = abs(res.uv - 0.5) * 2;
        float lmax = max(local.x, local.y);
        if (lmax > 1.0 - thickness) {
            color = vec4(1-color.rgb, 1);
            return;
        }
    }
}
