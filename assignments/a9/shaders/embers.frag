#version 330 core

in vec4 vtx_color;
out vec4 frag_color;

void main() {
    // Round particle
    vec2 coord = gl_PointCoord - vec2(0.5);
    float r = length(coord) * 2.0; // 0 to 1
    
    if (r > 1.0) discard;
    
    // Soft glow
    float glow = 1.0 - r;
    glow = pow(glow, 1.5);
    
    frag_color = vec4(vtx_color.rgb, vtx_color.a * glow);
}
