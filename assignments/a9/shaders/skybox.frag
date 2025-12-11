#version 330 core

uniform samplerCube skybox;
uniform float iTime;
in vec3 vtx_model_position;
out vec4 frag_color;

/* Hash function for 3D vector */
float hash(vec3 p) {
    p = fract(p * 0.3183099 + .1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

/* 3D Noise function */
float noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
                   mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
                   mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
}

/* Fragment Shader Main */
void main()
{
    vec3 dir = normalize(vtx_model_position);
    vec3 color = texture(skybox, dir).rgb;
    
    float gradient_start = 0.0;
    float gradient_end = 0.8;
    float opacity = 1.0 - smoothstep(gradient_start, gradient_end, dir.y);
    color *= opacity;

    float scale = 80.0;
    vec3 p = dir * scale;
    
    vec3 i = floor(p);
    float h = hash(i);
    
    float star = 0.0;
    if (h > 0.95) { 
        float twinkle = 0.5 + 0.5 * sin(iTime * 2.0 + h * 100.0);
        
        vec3 f = fract(p) - 0.5;
        float dist = length(f);
        float core = smoothstep(0.4, 0.0, dist);
        
        star = core * twinkle;
    }
    
    float brightness = dot(color, vec3(0.3, 0.59, 0.11));
    if(brightness < 0.2) {
        color += vec3(star);
    }

    frag_color = vec4(color, 1.0);
}
