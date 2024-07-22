#shader_type vertex
#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;

out vec2 v_uv;

void main() {
    v_uv = uv;
    gl_Position = vec4(position, 1.0);
}

// -------------------------------------------------------------------------------------------------

#shader_type fragment
#version 450

in vec2 v_uv;

layout (location = 0) out vec4 color;

uniform vec2 u_dimensions;

const float size = 4;

void main() {
    // convert uv into pixels
    vec2 coords = u_dimensions * v_uv;
    vec2 center = u_dimensions * 0.5;
    float dst = length(coords - center);

    // create smooth circle at center with radius of `size`
    float alpha = 1 - smoothstep(size, size+1, dst);
    color = vec4(vec3(1), alpha);
}
