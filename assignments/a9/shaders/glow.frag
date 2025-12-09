#version 330 core

in vec4 vtx_color;
uniform float time;
out vec4 frag_color;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float r = length(coord) * 2.0; 
    
    if (r > 1.0) discard;
    
    // Soft gaussian-like falloff
    float alpha = exp(-r*r*4.0);
    
    // Slight shimmer/pulse
    float pulse = 0.9 + 0.1 * sin(time * 3.0);
    
    frag_color = vec4(vtx_color.rgb, vtx_color.a * alpha * pulse);
}
