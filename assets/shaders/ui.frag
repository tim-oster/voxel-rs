#version 460

in vec2 v_uv;

layout (location = 0) out vec4 color;

uniform vec2 u_dimensions;

const float size = 4;

void main() {
    vec2 coords = u_dimensions * v_uv;
    vec2 center = u_dimensions * 0.5;
    float dst = length(coords - center);
    float alpha = 1 - smoothstep(size, size+1, dst);
    color = vec4(vec3(1), alpha);
}
