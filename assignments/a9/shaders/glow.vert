#version 330 core

layout(std140) uniform camera {
    mat4 projection;
    mat4 view;
    mat4 pvm;
    mat4 ortho;
    vec4 position;
};

layout(location = 0) in vec3 pos;

out vec4 vtx_color;

void main() {
    vec4 worldPos = vec4(pos, 1.0);
    gl_Position = pvm * worldPos; 
    
    // Size attenuation (larger than embers)
    float dist = length(worldPos.xyz - position.xyz);
    gl_PointSize = 800.0 / (dist + 0.1); 
    
    vtx_color = vec4(1.0, 0.6, 0.1, 0.3); // Default Orange base

    // Procedural Color Variation
    // Generate a random value based on Vertex ID (stable index for each particle)
    float seed = fract(sin(float(gl_VertexID) * 12.9898) * 43758.5453);
    
    // Logic matching bloom/lantern tints (Red, Green, Blue, Yellow)
    // We mix the base orange with the tint to keep it warm/glowing but colored.
    vec3 tint = vec3(1.0);
    if (seed < 0.25) tint = vec3(2.0, 0.2, 0.2); // Reddish
    else if (seed < 0.5) tint = vec3(0.2, 2.0, 0.2); // Greenish
    else if (seed < 0.75) tint = vec3(0.2, 0.5, 2.0); // Blueish
    else tint = vec3(1.5, 1.5, 0.2); // Yellowish
    
    // Mix or Multiply? Multiply gives "tinted orange".
    // Replacing entirely might be better to match the lantern faces.
    // Let's replace the RGB but keep alpha.
    
    vtx_color.rgb = tint;
}
