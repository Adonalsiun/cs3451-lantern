#version 330 core

layout(std140) uniform camera
{
    mat4 projection;
    mat4 view;
    mat4 pvm;
    mat4 ortho;
    vec4 position;
};

in vec3 pos;

out vec4 frag_color;
out float particle_alpha;

uniform float time;

void main()
{
    gl_Position = projection * view * vec4(pos, 1.0);
    gl_PointSize = 12.0;
    
    // We'll pass alpha through a uniform array or encode it somehow
    // For now, hard-code but we'll fix this
    frag_color = vec4(1.0, 0.7, 0.2, 1.0);
    particle_alpha = 1.0; // Will be overridden
}