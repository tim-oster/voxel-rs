#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;

out vec2 v_uv;

uniform mat4 view;
uniform mat4 projection;

void main() {
    v_uv = uv;
    gl_Position = projection * view * vec4(position, 1.0);
}
