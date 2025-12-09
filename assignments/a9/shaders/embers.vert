#version 330 core

layout(std140) uniform camera {
    mat4 projection;
    mat4 view;
    mat4 pvm;
    mat4 ortho;
    vec4 position;
};

layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 color;

out vec4 vtx_color;

void main() {
    vec4 worldPos = vec4(pos, 1.0);
    gl_Position = pvm * worldPos; // pvm includes view transform
    
    vtx_color = color;
    
    // Size attenuation
    float dist = length(worldPos.xyz - position.xyz);
    gl_PointSize = 300.0 / (dist + 0.1); 
}
