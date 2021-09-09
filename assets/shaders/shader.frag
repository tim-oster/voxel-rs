#version 460

in vec2 v_uv;
in vec3 v_normal;
in vec3 v_frag_pos;

layout (location = 0) out vec4 color;

uniform sampler2D u_texture;
uniform float u_ambient;
uniform vec3 u_ligth_dir;
uniform vec3 u_cam_pos;

const float specular_strength = 0.5;

void main() {
    float diffuse = max(dot(v_normal, -u_ligth_dir), 0.0);

    vec3 view_dir = normalize(v_frag_pos - u_cam_pos);
    vec3 reflect_dir = reflect(-u_ligth_dir, v_normal);
    float specular = pow(max(dot(view_dir, reflect_dir), 0.0), 256) * specular_strength;

    float light = u_ambient + diffuse + specular;
    color = texture(u_texture, v_uv) * light;
}
