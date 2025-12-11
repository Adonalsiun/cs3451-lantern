#version 330 core

/* default camera matrices. do not modify.*/
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
	vec4 pos; 
	vec4 dir;
	vec4 amb; 
	vec4 dif; 
	vec4 spec; 
	vec4 atten;
	vec4 r;
};
layout(std140) uniform lights
{
	vec4 amb;
	ivec4 lt_att; 
	light lt[4];
};

/* input variables */
in vec3 vtx_normal; 
in vec3 vtx_position; 
in vec3 vtx_model_position; 
in vec4 vtx_color;
in vec2 vtx_uv;
in vec3 vtx_tangent;

uniform vec3 ka;            
uniform vec3 kd;            
uniform vec3 ks;            
uniform float shininess;    

uniform sampler2D tex_color;   
uniform sampler2D tex_normal;   

uniform float time; 
uniform float seed; 

uniform int num_lanterns = 0; 
uniform vec3 lantern_positions[32]; 
uniform float global_shadow_factor = 0.5; 

/* output variables */
out vec4 frag_color;

/* Simple hash function */
float hash(float n) { return fract(sin(n) * 1e4); }
float hash(vec2 p) { return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x)))); }

/* 3D Noise function */
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

/* 2D noise function */
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

/* Reads normal from texture */
vec3 read_normal_texture()
{
    vec3 normal = texture(tex_normal, vtx_uv).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    return normal;
}

/* Fragment Shader Main */
void main()
{
    vec3 e = position.xyz;              
    vec3 p = vtx_position;              
    vec3 N = normalize(vtx_normal);     
    vec3 T = normalize(vtx_tangent);    
    vec3 B = normalize(cross(N, T));    
    mat3 TBN = mat3(T, B, N);

    vec3 texture_normal = read_normal_texture();
    vec3 world_normal = normalize(TBN * texture_normal);
    vec3 texture_color = texture(tex_color, vtx_uv).rgb;

    float flame_intensity = 0.0;
    
    vec2 flame_uv = vtx_uv * vec2(5.0, 2.0); 
    float n1 = noise(flame_uv + vec2(0.0, -time * 2.0));
    float n2 = noise(flame_uv * 2.0 + vec2(0.0, -time * 3.0));
    float flame_noise = n1 + n2 * 0.5;

    float hue_shift = fract(sin(seed) * 43758.5453); 
    vec3 tint = vec3(1.0);
    if (hue_shift < 0.25) tint = vec3(2.0, 0.5, 0.5); 
    else if (hue_shift < 0.5) tint = vec3(0.5, 2.0, 0.5); 
    else if (hue_shift < 0.75) tint = vec3(0.5, 0.5, 2.0); 
    else tint = vec3(2.0, 2.0, 0.5); 
    
    texture_color *= tint;
    
    vec2 temp_uv = vtx_uv; 
    float pattern_type = fract(sin(seed * 12.9898) * 43758.5453); 
    float pattern = 1.0;
    
    if (pattern_type < 0.3) {
        float stripes = sin(temp_uv.x * 50.0 + seed); 
        pattern = 0.5 + 0.5 * stripes; 
    } else if (pattern_type < 0.6) {
        vec2 grid = fract(temp_uv * 15.0) - 0.5;
        float dist = length(grid);
        if (dist < 0.3) pattern = 1.5; else pattern = 0.5; 
    } else {
        float n = noise(vtx_uv * 10.0 + seed); 
        pattern = 0.5 + 1.0 * n; 
    }
    
    texture_color *= pattern;
    
    float pulse = 0.8 + 0.2 * sin(time * 5.0 + noise(vec2(time)) * 10.0);
    
    float is_paper = smoothstep(0.2, 0.4, length(texture_color)); 
    
    vec3 flame_color = vec3(1.0, 0.6, 0.1) * tint; 
    vec3 inner_light = flame_color * pulse * 2.0;

    inner_light *= (0.8 + 0.4 * flame_noise);

    vec3 frame_lighting = vec3(0.0);
    vec3 ambient = ka * 0.2; 
    vec3 V = normalize(e - p);

    for(int i=0; i<lt_att[0] && i < 4; ++i) {
        vec3 L = normalize(lt[i].pos.xyz - p);
        if(lt[i].pos.w == 0.0) L = normalize(lt[i].dir.xyz); 
        
        float diff = max(dot(world_normal, L), 0.0);
        vec3 diffuse = kd * diff * lt[i].dif.rgb;
        
        vec3 R = reflect(-L, world_normal);
        float spec = pow(max(dot(V, R), 0.0), shininess);
        vec3 specular = ks * spec * lt[i].spec.rgb;
        
        float attenuation = 1.0;
        if(lt[i].pos.w != 0.0) {
            float dist = length(lt[i].pos.xyz - p);
            attenuation = 1.0 / (lt[i].atten.x + lt[i].atten.y * dist + lt[i].atten.z * dist * dist);
        }
        
        frame_lighting += (diffuse + specular) * attenuation; 
    }
    
    vec3 lantern_lighting = vec3(0.0);
    vec3 lantern_color = vec3(1.0, 0.6, 0.2); 
    float lantern_intensity = 2.0;
    
    for(int i = 0; i < num_lanterns && i < 32; ++i) {
        vec3 lightPos = lantern_positions[i];
        vec3 L = normalize(lightPos - p);
        float dist = length(lightPos - p);
        
        float shadowFactor = 1.0;

        int maxChecks = (num_lanterns < 10) ? num_lanterns : 10; 
        
        for(int j = 0; j < maxChecks && j < 32; ++j) {
            if (j == i) continue; 
            
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
                
                if (alignment > 0.7) {
                    float occlusion = 1.0 - smoothstep(0.0, distToLight, distToOther);
                    occlusion *= smoothstep(0.7, 1.0, alignment);
                    shadowFactor = min(shadowFactor, 1.0 - occlusion * 0.95); 
                    
                    if (shadowFactor < 0.1) break;
                }
            }
        }
        
        shadowFactor = min(shadowFactor, global_shadow_factor);
        
        shadowFactor = mix(0.0, 1.0, shadowFactor);
        
        if (shadowFactor < 0.3) {
            shadowFactor *= 0.3;
        }
        
        shadowFactor = pow(shadowFactor, 1.5); 
        
        float diff = max(dot(world_normal, L), 0.0);
        vec3 diffuse = kd * diff * lantern_color * lantern_intensity;
        
        vec3 R = reflect(-L, world_normal);
        float spec = pow(max(dot(V, R), 0.0), shininess);
        vec3 specular = ks * spec * lantern_color * lantern_intensity;
        
        float minDist = 0.5;
        float attenDist = max(dist, minDist);
        float attenuation = 1.0 / (1.0 + 0.1 * attenDist + 0.01 * attenDist * attenDist);
        
        lantern_lighting += (diffuse + specular) * attenuation * shadowFactor;
    }
    
    frame_lighting += ambient + lantern_lighting;
    
    vec3 final_color = texture_color * frame_lighting;
    
    final_color += inner_light * is_paper * 0.8; 
    
    float rim = 1.0 - max(dot(world_normal, V), 0.0);
    rim = pow(rim, 2.5); 

    vec3 rim_color = tint * vec3(1.0, 0.6, 0.2); 
    float rim_intensity = is_paper * 1.5 + 0.3; 

    float rim_pulse = 0.8 + 0.2 * sin(time * 3.0 + seed);
    rim_intensity *= rim_pulse;

    final_color += rim_color * rim * rim_intensity;

    float edge_highlight = pow(rim, 8.0); 
    final_color += vec3(1.0, 0.8, 0.5) * edge_highlight * 0.5; 
    
    frag_color = vec4(final_color, 1.0);
}
