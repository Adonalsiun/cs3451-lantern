#version 330 core

in vec4 frag_color;
in float particle_alpha;

out vec4 out_color;

uniform float particle_alphas[2000]; // Array of alpha values for each particle

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    if (dist > 0.5) discard;
    
    // Soft falloff
    float soft_edge = (1.0 - smoothstep(0.2, 0.5, dist));
    
    // Use the alpha from the uniform array (passed per-particle)
    float final_alpha = soft_edge * 0.8 * particle_alpha;
    
    out_color = vec4(1.0, 0.7, 0.2, final_alpha);
}