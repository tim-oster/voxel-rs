#version 460

in vec2 v_uv;
in vec3 v_normal;
in vec3 v_frag_pos;

layout (location = 0) out vec4 color;

uniform sampler2D u_texture;
uniform float u_ambient;
uniform vec3 u_light_dir;
uniform vec3 u_cam_pos;

const float specular_strength = 0.5;

// https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Buffer_backed
// https://www.khronos.org/opengl/wiki/Shader_Storage_Buffer_Object
// https://www.iquilezles.org/www/articles/boxfunctions/boxfunctions.htm
layout (std430, binding = 0) buffer root_node {
    vec3 position;
    int descriptors[];
};

void main() {
    float diffuse = max(dot(v_normal, -u_light_dir), 0.0);

    vec3 view_dir = normalize(v_frag_pos - u_cam_pos);
    vec3 reflect_dir = reflect(-u_light_dir, v_normal);
    float specular = pow(max(dot(view_dir, reflect_dir), 0.0), 256) * specular_strength;

//    uint child_pointer = (descriptors[0] & 0xffff);
//    uint valid_mask = (descriptors[0] & 0x0000ff);
//    uint child_mask = (descriptors[0] & 0x000000ff);

    float light = u_ambient + diffuse + specular;
    color = texture(u_texture, v_uv) * light;
}
