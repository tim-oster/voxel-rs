#version 460

in vec2 v_uv;

layout (location = 0) out vec4 color;

uniform mat4 u_view;
uniform float u_fovy;
uniform float u_aspect;

const float specular_strength = 0.5;

// https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Buffer_backed
// https://www.khronos.org/opengl/wiki/Shader_Storage_Buffer_Object
layout (std430, binding = 0) buffer root_node {
    vec3 position;
    int descriptors[];
};

// source: https://www.iquilezles.org/www/articles/boxfunctions/boxfunctions.htm
vec2 boxIntersection(in vec3 ro, in vec3 rd, in vec3 rad, out vec3 oN) {
    vec3 m = 1.0/rd;
    vec3 n = m*ro;
    vec3 k = abs(m)*rad;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

    float tN = max(max(t1.x, t1.y), t1.z);
    float tF = min(min(t2.x, t2.y), t2.z);

    if (tN>tF || tF<0.0) return vec2(-1.0);// no intersection

    oN = -sign(rd)*step(t1.yzx, t1.xyz)*step(t1.zxy, t1.xyz);

    return vec2(tN, tF);
}

float sphIntersect(vec3 ro, vec3 rd, vec4 sph) {
    vec3 oc = ro - sph.xyz;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - sph.w*sph.w;
    float h = b*b - c;
    if (h<0.0) return -1.0;
    h = sqrt(h);
    return -b - h;
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

    float d = sphIntersect(ro, rd, vec4(0.0, 0.0, -4.0, 1.0));

    color = vec4(d, 0.0, 0.0, 1.0);
}
