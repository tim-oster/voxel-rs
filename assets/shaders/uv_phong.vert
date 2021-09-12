#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;
layout (location = 2) in vec3 normal;

out vec2 v_uv;
out vec3 v_normal;
out vec3 v_frag_pos;

uniform mat4 u_view;
uniform mat4 u_projection;

void main() {
    v_uv = uv;
    v_normal = normal;
    v_frag_pos = position;
    gl_Position = u_projection * u_view * vec4(position, 1.0);
}
