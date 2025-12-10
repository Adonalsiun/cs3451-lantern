#version 330 core

in vec3 frag_position;
in vec3 frag_normal;
in vec2 frag_uv;

out vec4 frag_color;

uniform float time;
uniform vec3 camera_position;

void main()
{
    // Animated water color
    vec3 deep_water = vec3(0.0, 0.1, 0.2);
    vec3 shallow_water = vec3(0.1, 0.3, 0.4);
    
    // Create flowing pattern
    float pattern = sin(frag_uv.x * 10.0 + time) * 
                    cos(frag_uv.y * 10.0 + time * 0.8) * 0.5 + 0.5;
    
    vec3 water_color = mix(deep_water, shallow_water, pattern);
    
    // Add some reflectivity based on view angle
    vec3 view_dir = normalize(camera_position - frag_position);
    float fresnel = pow(1.0 - max(dot(view_dir, vec3(0, 1, 0)), 0.0), 3.0);
    
    // Mix in some brightness for reflection
    water_color += vec3(0.1, 0.15, 0.2) * fresnel;
    
    // Add lantern glow reflection (orange tint)
    water_color += vec3(0.3, 0.15, 0.05) * pattern * 0.3;
    
    frag_color = vec4(water_color, 0.7); // Semi-transparent
}