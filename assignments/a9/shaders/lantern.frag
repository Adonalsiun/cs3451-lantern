#version 330 core

/*default camera matrices. do not modify.*/
layout(std140) uniform camera
{
    mat4 projection;	/*camera's projection matrix*/
    mat4 view;			/*camera's view matrix*/
    mat4 pvm;			/*camera's projection*view*model matrix*/
    mat4 ortho;			/*camera's ortho projection matrix*/
    vec4 position;		/*camera's position in world space*/
};

/* set light ubo. do not modify.*/
struct light
{
	ivec4 att; 
	vec4 pos; // position
	vec4 dir;
	vec4 amb; // ambient intensity
	vec4 dif; // diffuse intensity
	vec4 spec; // specular intensity
	vec4 atten;
	vec4 r;
};
layout(std140) uniform lights
{
	vec4 amb;
	ivec4 lt_att; // lt_att[0] = number of lights
	light lt[4];
};

/*input variables*/
in vec3 vtx_normal; // vtx normal in world space
in vec3 vtx_position; // vtx position in world space
in vec3 vtx_model_position; // vtx position in model space
in vec4 vtx_color;
in vec2 vtx_uv;
in vec3 vtx_tangent;

uniform vec3 ka;            /* object material ambient */
uniform vec3 kd;            /* object material diffuse */
uniform vec3 ks;            /* object material specular */
uniform float shininess;    /* object material shininess */

uniform sampler2D tex_color;   /* texture sampler for color */
uniform sampler2D tex_normal;   /* texture sampler for normal vector */

uniform float time; // Time for animation
uniform float seed; // PCG Seed

// Shadow computation uniforms
uniform int num_lanterns = 0; // Number of lantern light sources
uniform vec3 lantern_positions[32]; // Lantern positions (max 32)
uniform float global_shadow_factor = 0.5; // Global shadow factor (0.0 = shadowed, 1.0 = lit)

/*output variables*/
out vec4 frag_color;

// Simple noise function
float hash(float n) { return fract(sin(n) * 1e4); }
float hash(vec2 p) { return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x)))); }

float noise(vec3 x) {
    const vec3 step = vec3(110, 241, 171);
    vec3 i = floor(x);
    vec3 f = fract(x);
    float n = dot(i, step);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
               mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}

// 2D noise for flame
float noise(vec2 x) {
    vec2 i = floor(x);
    vec2 f = fract(x);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

vec3 read_normal_texture()
{
    vec3 normal = texture(tex_normal, vtx_uv).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    return normal;
}

void main()
{
    vec3 e = position.xyz;              //// eye position
    vec3 p = vtx_position;              //// surface position
    vec3 N = normalize(vtx_normal);     //// normal vector
    vec3 T = normalize(vtx_tangent);    //// tangent vector
    vec3 B = normalize(cross(N, T));    //// binormal vector
    mat3 TBN = mat3(T, B, N);

    vec3 texture_normal = read_normal_texture();
    vec3 world_normal = normalize(TBN * texture_normal);
    vec3 texture_color = texture(tex_color, vtx_uv).rgb;

    // --- Procedural Flame Effect ---
    // Assuming the "flame" part is the paper itself or center.
    // Since the whole lantern uses the same shader, we might need a mask or assume the paper is the flame bearer.
    // The lantern texture likely has the paper frame. Let's assume the paper is the main emissive part.
    // We can use the brightness of the texture_color to determine if it's paper (bright) or frame (dark/black).
    
    // Better yet, we can use the UV coordinates. A typical lantern UV map unwraps the cylinder.
    
    float flame_intensity = 0.0;
    
    // Let's create a procedural flame pattern
    vec2 flame_uv = vtx_uv * vec2(5.0, 2.0); // Scale UV
    float n1 = noise(flame_uv + vec2(0.0, -time * 2.0));
    float n2 = noise(flame_uv * 2.0 + vec2(0.0, -time * 3.0));
    float flame_noise = n1 + n2 * 0.5;

    // --- Procedural Patterns (PCG) ---
    // Use seed to determine pattern type and color shift
    
    // 1. Color Shift
    float hue_shift = fract(sin(seed) * 43758.5453); // 0-1 random
    vec3 tint = vec3(1.0);
    if (hue_shift < 0.25) tint = vec3(2.0, 0.5, 0.5); // Strong Red
    else if (hue_shift < 0.5) tint = vec3(0.5, 2.0, 0.5); // Strong Green
    else if (hue_shift < 0.75) tint = vec3(0.5, 0.5, 2.0); // Strong Blue
    else tint = vec3(2.0, 2.0, 0.5); // Strong Yellow
    
    texture_color *= tint;
    
    // 2. Pattern
    vec2 temp_uv = vtx_uv; 
    float pattern_type = fract(sin(seed * 12.9898) * 43758.5453); 
    float pattern = 1.0;
    
    if (pattern_type < 0.3) {
        // Stripes
        float stripes = sin(temp_uv.x * 50.0 + seed); // Higher frequency
        pattern = 0.5 + 0.5 * stripes; // High contrast
    } else if (pattern_type < 0.6) {
        // Polka dots
        vec2 grid = fract(temp_uv * 15.0) - 0.5;
        float dist = length(grid);
        if (dist < 0.3) pattern = 1.5; else pattern = 0.5; // High contrast dots
    } else {
        // Noise patch
        float n = noise(vtx_uv * 10.0 + seed); // 2D noise
        pattern = 0.5 + 1.0 * n; // High contrast noise
    }
    
    texture_color *= pattern;
    // --- End PCG ---
    // float flame_noise = n1 + n2 * 0.5; // Removed duplicate definition
    
    // Basic inner glow (pulsing)
    float pulse = 0.8 + 0.2 * sin(time * 5.0 + noise(vec2(time)) * 10.0);
    
    // Paper transparency / translucency
    // We treat the paper as translucent. Light from inside (point light) should illuminate it powerfully.
    // Since we don't have true subsurface scattering, we fake it with emissive color.
    
    // Using texture brightness as a mask for "Paper" vs "Frame"
    float is_paper = smoothstep(0.2, 0.4, length(texture_color)); 
    
    vec3 flame_color = vec3(1.0, 0.6, 0.1) * tint; // Warm orange mixed with tint
    vec3 inner_light = flame_color * pulse * 2.0;

    // Add noise to the inner light to simulate fire moving inside
    inner_light *= (0.8 + 0.4 * flame_noise);

    // --- Lighting (Phong) ---
    // Light 0 is warm point light from scene (sun/moon/lamp). 
    // Light 1 is "inner" light?? No, usually scene lights.
    // We iterate over lights.
    
    vec3 frame_lighting = vec3(0.0);
    vec3 ambient = ka * 0.2; // global ambient
    vec3 V = normalize(e - p);

    for(int i=0; i<lt_att[0] && i < 4; ++i) {
        vec3 L = normalize(lt[i].pos.xyz - p);
        if(lt[i].pos.w == 0.0) L = normalize(lt[i].dir.xyz); // Directional
        
        // Diffuse
        float diff = max(dot(world_normal, L), 0.0);
        vec3 diffuse = kd * diff * lt[i].dif.rgb;
        
        // Specular
        vec3 R = reflect(-L, world_normal);
        float spec = pow(max(dot(V, R), 0.0), shininess);
        vec3 specular = ks * spec * lt[i].spec.rgb;
        
        // Attenuation
        float attenuation = 1.0;
        if(lt[i].pos.w != 0.0) {
            float dist = length(lt[i].pos.xyz - p);
            attenuation = 1.0 / (lt[i].atten.x + lt[i].atten.y * dist + lt[i].atten.z * dist * dist);
        }
        
        frame_lighting += (diffuse + specular) * attenuation; // ambient added once
    }
    
    // --- Lantern Lighting with Shadow Computation ---
    // Add lighting from lanterns (light sources) with shadow factors
    // Note: Full shadow ray tracing is computed on CPU, but we can add
    // per-lantern lighting here. For full shadows, shadow factors would
    // be passed as vertex attributes or computed per-vertex.
    
    vec3 lantern_lighting = vec3(0.0);
    vec3 lantern_color = vec3(1.0, 0.6, 0.2); // Warm orange lantern light
    float lantern_intensity = 2.0;
    
    for(int i = 0; i < num_lanterns && i < 32; ++i) {
        vec3 lightPos = lantern_positions[i];
        vec3 L = normalize(lightPos - p);
        float dist = length(lightPos - p);
        

        float shadowFactor = 1.0;

        int maxChecks = (num_lanterns < 10) ? num_lanterns : 10; // Limit to 10 lanterns max -- debugging
        
        // For each other lantern, check if it's between this fragment and the current light
        for(int j = 0; j < maxChecks && j < 32; ++j) {
            if (j == i) continue; // Skip the current light source
            
            vec3 otherLanternPos = lantern_positions[j];
            vec3 toOtherLantern = otherLanternPos - p;
            vec3 toLight = lightPos - p;
            

            float distToOther = length(toOtherLantern);
            float distToLight = length(toLight);

            if (distToOther > distToLight * 2.0) continue;
            
            if (distToOther < distToLight) {
                vec3 dirToOther = normalize(toOtherLantern);
                vec3 dirToLight = normalize(toLight);
                float alignment = dot(dirToOther, dirToLight);
                
                // If lanterns are aligned (other lantern is between fragment and light)
                if (alignment > 0.7) {
                    float occlusion = 1.0 - smoothstep(0.0, distToLight, distToOther);
                    // Larger occlusion for better aligned lanterns
                    occlusion *= smoothstep(0.7, 1.0, alignment);
                    shadowFactor = min(shadowFactor, 1.0 - occlusion * 0.95); // VERY strong occlusion (95%)
                    
                    if (shadowFactor < 0.1) break;
                }
            }
        }
        
        // global shadow factor as a base
        shadowFactor = min(shadowFactor, global_shadow_factor);
        
        // Make shadows EXTREMELY dark almost completely black when shadowed
        shadowFactor = mix(0.0, 1.0, shadowFactor);
        
        if (shadowFactor < 0.3) {
            shadowFactor *= 0.3;
        }
        
        // Apply exponential darkening for more dramatic effect
        shadowFactor = pow(shadowFactor, 1.5); 
        
        // Diffuse lighting from lantern
        float diff = max(dot(world_normal, L), 0.0);
        vec3 diffuse = kd * diff * lantern_color * lantern_intensity;
        
        // Specular
        vec3 R = reflect(-L, world_normal);
        float spec = pow(max(dot(V, R), 0.0), shininess);
        vec3 specular = ks * spec * lantern_color * lantern_intensity;
        
        // Distance attenuation (inverse square law with minimum distance)
        float minDist = 0.5;
        float attenDist = max(dist, minDist);
        float attenuation = 1.0 / (1.0 + 0.1 * attenDist + 0.01 * attenDist * attenDist);
        
        // Apply shadow factor
        lantern_lighting += (diffuse + specular) * attenuation * shadowFactor;
    }
    
    frame_lighting += ambient + lantern_lighting;
    
    // Combine
    // If paper, use emissive mainly. If frame, use lighting.
    // mix based on is_paper?
    // is_paper is 1.0 for paper, 0.0 for frame.
    // Actually the texture has dark frame. 
    // Let's add lighting to EVERYTHING, but Paper adds Emission.
    
    vec3 final_color = texture_color * frame_lighting;
    
    // Add emission
    final_color += inner_light * is_paper * 0.8; 
    
    // --- Volumetric Glow (Fake Halo) ---
    // Calculate rim light or view-dependent glow
    // vec3 V is already defined
    float rim = 1.0 - max(dot(world_normal, V), 0.0);
    rim = pow(rim, 3.0);
    
    final_color += vec3(1.0, 0.5, 0.0) * rim * is_paper * 0.5;

    frag_color = vec4(final_color, 1.0);
}
