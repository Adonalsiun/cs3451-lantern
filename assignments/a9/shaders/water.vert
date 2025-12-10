#version 330 core

layout(location = 0) in vec3 vtx_position;
layout(location = 1) in vec3 vtx_normal;
layout(location = 2) in vec2 vtx_uv;

out vec3 frag_position;
out vec3 frag_normal;
out vec2 frag_uv;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform float time;

void main()
{
    vec3 pos = vtx_position;
    
    // Create wave effect
    float wave1 = sin(pos.x * 0.5 + time * 2.0) * 0.3;
    float wave2 = sin(pos.z * 0.3 + time * 1.5) * 0.2;
    pos.y += wave1 + wave2;
    
    frag_position = vec3(model * vec4(pos, 1.0));
    frag_normal = mat3(transpose(inverse(model))) * vtx_normal;
    frag_uv = vtx_uv;
    
    gl_Position = projection * view * vec4(frag_position, 1.0);
}