#include "Common.h"
#include "OpenGLCommon.h"
#include "OpenGLMarkerObjects.h"
#include "OpenGLParticles.h"
#include "OpenGLBgEffect.h"
#include "OpenGLMesh.h"
#include "OpenGLViewer.h"
#include "OpenGLWindow.h"
#include "TinyObjLoader.h"
#include "OpenGLSkybox.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>
#include <string>
#include <cmath>

#ifndef __Main_cpp__
#define __Main_cpp__

#ifdef __APPLE__
#define CLOCKS_PER_SEC 100000
#endif

// Custom Lantern Object for PCG
class LanternObject : public OpenGLTriangleMesh {
public:
    float seed = 0.0f;
    // AABB for shadow optimization
    Vector3 min_aabb;
    Vector3 max_aabb;

    virtual void Set_Shader_Parameters() const override {
        // Pass seed to shader
        if (shader_programs.size() > 0) {
            shader_programs[0]->Set_Uniform("seed", seed);
        }
    }
};

class MyDriver : public OpenGLViewer
{
    std::vector<OpenGLTriangleMesh *> mesh_object_array;
    OpenGLBgEffect *bgEffect = nullptr;
    OpenGLSkybox *skybox = nullptr;
    // OpenGLParticles<Particles<3>>* embers = nullptr; // Removed
    OpenGLParticles<Particles<3>>* glow_system = nullptr; // Glow System
    OpenGLParticles<Particles<3>>* trail_system = nullptr;
    clock_t startTime;

    OpenGLTriangleMesh* water_plane = nullptr;

    // Lanterns
    // Lanterns
    LanternObject* hero_lantern = nullptr;
    std::vector<LanternObject*> background_lanterns;

    // Add to MyDriver class private section
    struct TrailParticle {
        int index;          // Index in particle array
        float birth_time;   // When particle was created
        Vector3 velocity;   // Particle velocity
    };

    std::vector<TrailParticle> active_trails;
    int next_trail_index = 0;
    float last_trail_spawn_time = 0.0f;

public:
    virtual void Initialize()
    {
        draw_axes = false;
        startTime = clock();
        OpenGLViewer::Initialize();
        
        // Initial Camera Setup
        if (opengl_window) {
            opengl_window->camera_distance = 15.0f;
            opengl_window->camera_target = Vector3f(0, 5, 0);
            opengl_window->rotation_matrix.setIdentity();
        }
    }

    virtual void Initialize_Data()
    {
        //// Load Shaders
        OpenGLShaderLibrary::Instance()->Add_Shader_From_File("shaders/basic.vert", "shaders/basic.frag", "basic");
        OpenGLShaderLibrary::Instance()->Add_Shader_From_File("shaders/lantern.vert", "shaders/lantern.frag", "lantern"); 
        // Embers shader removed
        OpenGLShaderLibrary::Instance()->Add_Shader_From_File("shaders/glow.vert", "shaders/glow.frag", "glow");
        OpenGLShaderLibrary::Instance()->Add_Shader_From_File("shaders/skybox.vert", "shaders/skybox.frag", "skybox");

        OpenGLShaderLibrary::Instance()->Add_Shader_From_File("shaders/water.vert", 
                                                       "shaders/water.frag", 
                                                       "water");

        OpenGLShaderLibrary::Instance()->Add_Shader_From_File("shaders/trail.vert", 
                                                        "shaders/trail.frag", 
                                                        "trail");
        
        //// Load Textures
        OpenGLTextureLibrary::Instance()->Add_Texture_From_File("tex/bunny_color.jpg", "bunny_color");
        OpenGLTextureLibrary::Instance()->Add_Texture_From_File("tex/bunny_normal.png", "bunny_normal");
        OpenGLTextureLibrary::Instance()->Add_Texture_From_File("tex/lantern_color.png", "lantern_color");
        OpenGLTextureLibrary::Instance()->Add_Texture_From_File("tex/lantern_normal.png", "lantern_normal");
        OpenGLTextureLibrary::Instance()->Add_Texture_From_File("tex/buzz_color.png", "buzz_color"); 

        //// Add Lights
        opengl_window->Add_Light(Vector3f(3, 1, 3), Vector3f(0.2, 0.2, 0.2), Vector3f(0.8, 0.5, 0.2), Vector3f(0.5, 0.5, 0.5)); // Warm light
        opengl_window->Add_Light(Vector3f(0, 0, -5), Vector3f(0.1, 0.1, 0.1), Vector3f(0.9, 0.9, 0.9), Vector3f(0.5, 0.5, 0.5));

        //// Sky box
        {
            const std::vector<std::string> cubemap_files{
                "cubemap/posx.jpg", "cubemap/negx.jpg",
                "cubemap/posy.jpg", "cubemap/negy.jpg",
                "cubemap/posz.jpg", "cubemap/negz.jpg", 
            };
            OpenGLTextureLibrary::Instance()->Add_CubeMap_From_Files(cubemap_files, "cube_map");

            skybox = Add_Interactive_Object<OpenGLSkybox>();
            skybox->Add_Shader_Program(OpenGLShaderLibrary::Get_Shader("skybox"));
            skybox->Initialize();
        }

        //// Embers - Removed
        /*
        {
            embers = Add_Interactive_Object<OpenGLParticles<Particles<3>>>();
            ...
        }
        */

        //// Glow System
        {
            glow_system = Add_Interactive_Object<OpenGLParticles<Particles<3>>>();
            glow_system->particles.Add_Elements(1 + 100);
            glow_system->Initialize();
            
            // Fix: Initialize points and Override shader
            glow_system->opengl_points.Initialize();
            glow_system->opengl_points.shader_programs[0] = OpenGLShaderLibrary::Get_Shader("glow");
            glow_system->Set_Color(OpenGLColor(1.0f, 0.5f, 0.0f, 1.0f)); // Orange glow
        }

        {
            int max_trail_particles = 2000;
            trail_system = Add_Interactive_Object<OpenGLParticles<Particles<3>>>();
            trail_system->particles.Add_Elements(max_trail_particles);
            trail_system->Initialize();
            
            trail_system->opengl_points.Initialize();
            trail_system->opengl_points.shader_programs[0] = OpenGLShaderLibrary::Get_Shader("trail");
            trail_system->Set_Color(OpenGLColor(1.0f, 0.5f, 0.0f, 0.0f)); // Start transparent
        }

        //// Water Plane
        {
            water_plane = Create_Water_Plane(50.0f, 50); // 50x50 size, 50 subdivisions
            
            // Position at y = -10 (below lanterns)
            Matrix4f t;
            t << 1, 0, 0, 0,
                0, 1, 0, -10,
                0, 0, 1, 0,
                0, 0, 0, 1;
            water_plane->Set_Model_Matrix(t);
            
            // Set material properties
            water_plane->Set_Ka(Vector3f(0.1, 0.1, 0.2));
            water_plane->Set_Kd(Vector3f(0.2, 0.4, 0.5));
            water_plane->Set_Ks(Vector3f(0.8, 0.8, 0.8));
            water_plane->Set_Shininess(128);
            
            water_plane->Add_Shader_Program(OpenGLShaderLibrary::Get_Shader("water"));
            Set_Polygon_Mode(water_plane, PolygonMode::Fill);
            Set_Shading_Mode(water_plane, ShadingMode::TexAlpha);
            water_plane->Set_Data_Refreshed();
            water_plane->Initialize();
        }

        //// Hero Lantern
        {
            hero_lantern = Add_Lantern_Object("obj/JapaneseLamp.obj", 123.4f);
            Matrix4f t;
            t << 0.5, 0, 0, 0,
                 0, 0.5, 0, -5, // Start low
                 0, 0, 0.5, 0,
                 0, 0, 0, 1;
            hero_lantern->Set_Model_Matrix(t);

            hero_lantern->Set_Ka(Vector3f(0.2, 0.2, 0.2)); 
            hero_lantern->Set_Kd(Vector3f(0.8, 0.8, 0.8)); 
            hero_lantern->Set_Ks(Vector3f(1, 1, 1));
            hero_lantern->Set_Shininess(64);

            hero_lantern->Add_Texture("tex_color", OpenGLTextureLibrary::Get_Texture("lantern_color"));
            hero_lantern->Add_Texture("tex_normal", OpenGLTextureLibrary::Get_Texture("lantern_normal"));
            hero_lantern->Add_Shader_Program(OpenGLShaderLibrary::Get_Shader("lantern")); 
        }

        //// Background Lanterns
        int num_lanterns = 100;
        std::mt19937 rng(42); 
        std::uniform_real_distribution<float> dist_x(-40.0f, 40.0f);
        std::uniform_real_distribution<float> dist_y(-8.0f, 20.0f);
        std::uniform_real_distribution<float> dist_z(-30.0f, 30.0f);
        std::uniform_real_distribution<float> dist_scale(0.8f, 1.8f); 

        for (int i = 0; i < num_lanterns; ++i) {
            float seed = (float)i * 17.5f + 42.0f;
            auto lantern = Add_Lantern_Object("obj/JapaneseLamp.obj", seed);
            background_lanterns.push_back(lantern);

            float s = dist_scale(rng);
            float x = dist_x(rng);
            float y = dist_y(rng);
            float z = dist_z(rng);

            if (std::abs(x) < 2 && std::abs(y + 5) < 2) x += 5; // Avoid hero

            Matrix4f t;
            t << s, 0, 0, x,
                 0, s, 0, y,
                 0, 0, s, z,
                 0, 0, 0, 1;
            lantern->Set_Model_Matrix(t);

            lantern->Set_Ka(Vector3f(0.2, 0.2, 0.2));
            lantern->Set_Kd(Vector3f(0.8, 0.8, 0.8));
            lantern->Set_Ks(Vector3f(1, 1, 1));
            lantern->Set_Shininess(64);

            lantern->Add_Texture("tex_color", OpenGLTextureLibrary::Get_Texture("lantern_color"));
            lantern->Add_Texture("tex_normal", OpenGLTextureLibrary::Get_Texture("lantern_normal")); 
            lantern->Add_Shader_Program(OpenGLShaderLibrary::Get_Shader("lantern"));
        }


        //// Initialize rendering model
        for (auto &mesh_obj : mesh_object_array){
            Set_Polygon_Mode(mesh_obj, PolygonMode::Fill);
            Set_Shading_Mode(mesh_obj, ShadingMode::TexAlpha);
            mesh_obj->Set_Data_Refreshed();
            mesh_obj->Initialize();
        }
        Toggle_Play();
    }

    LanternObject *Add_Lantern_Object(std::string obj_file_name, float seed)
    {
        auto mesh_obj = Add_Interactive_Object<LanternObject>();
        Array<std::shared_ptr<TriangleMesh<3>>> meshes;
        Obj::Read_From_Obj_File_Discrete_Triangles(obj_file_name, meshes);

        mesh_obj->mesh = *meshes[0];
        mesh_obj->seed = seed;
        
        // Compute AABB
        if (!mesh_obj->mesh.Vertices().empty()) {
            Vector3 minB = mesh_obj->mesh.Vertices()[0];
            Vector3 maxB = mesh_obj->mesh.Vertices()[0];
            for (const auto& v : mesh_obj->mesh.Vertices()) {
                minB = minB.cwiseMin(v);
                maxB = maxB.cwiseMax(v);
            }
            mesh_obj->min_aabb = minB;
            mesh_obj->max_aabb = maxB;
        }

        mesh_object_array.push_back(mesh_obj);
        return mesh_obj;
    }

    OpenGLTriangleMesh* Create_Water_Plane(float size, int subdivisions)
    {
        auto mesh_obj = Add_Interactive_Object<OpenGLTriangleMesh>();
        
        // Create vertices for a subdivided plane
        std::vector<Vector3> vertices;
        std::vector<Vector3i> triangles;
        std::vector<Vector2> uvs;
        
        float step = size / subdivisions;
        
        // Generate vertices
        for (int z = 0; z <= subdivisions; ++z) {
            for (int x = 0; x <= subdivisions; ++x) {
                float px = -size/2 + x * step;
                float pz = -size/2 + z * step;
                vertices.push_back(Vector3(px, 0, pz));
                
                // UV coordinates
                uvs.push_back(Vector2(float(x)/subdivisions, float(z)/subdivisions));
            }
        }
        
        // Generate triangles
        for (int z = 0; z < subdivisions; ++z) {
            for (int x = 0; x < subdivisions; ++x) {
                int i0 = z * (subdivisions + 1) + x;
                int i1 = i0 + 1;
                int i2 = i0 + (subdivisions + 1);
                int i3 = i2 + 1;
                
                // Two triangles per quad
                triangles.push_back(Vector3i(i0, i2, i1));
                triangles.push_back(Vector3i(i1, i2, i3));
            }
        }
        
        // Set mesh data
        mesh_obj->mesh.Vertices() = vertices;
        mesh_obj->mesh.Elements() = triangles;
        
        // Calculate normals (all pointing up for a flat plane)
        std::vector<Vector3> normals(vertices.size(), Vector3(0, 1, 0));
        mesh_obj->mesh.Normals() = normals;
        
        // Set UVs
        mesh_obj->mesh.Uvs() = uvs;
 
        return mesh_obj;
    }

    // ========== Shadow Ray Computation Functions ==========
    //
    // These functions implement shadow computation by casting shadow rays from intersection
    // points to light sources (lanterns). The implementation uses ray-triangle intersection
    // testing to determine if a point is occluded from a light source.
    //
    // Usage Example:
    //   Vector3f intersectionPoint(0.0f, 5.0f, 0.0f);  // World space point
    //   float shadowFactor = ComputeShadowFactor(intersectionPoint);
    //   // shadowFactor is 0.0 (fully shadowed) to 1.0 (fully lit)
    //
    //   // Or check individual lights:
    //   std::vector<Vector3f> lanterns = GetLanternPositions();
    //   for (const auto& lightPos : lanterns) {
    //       bool inShadow = IsPointInShadow(intersectionPoint, lightPos);
    //       // Use inShadow to modulate lighting contribution
    //   }
    //
    
    /**
     * Transform a 3D point from model space to world space using a 4x4 transformation matrix
     */
    Vector3f TransformPoint(const Matrix4f& transform, const Vector3& point)
    {
        Vector4f homogeneous(static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2]), 1.0f);
        Vector4f transformed = transform * homogeneous;
        return Vector3f(transformed[0], transformed[1], transformed[2]);
    }

    /**
     * Ray-Triangle Intersection TEST ONLY (Moller-Trumbore algorithm)
     * Returns true if ray intersects triangle, and stores the intersection distance in t
     */
    bool RayTriangleIntersect(
        const Vector3f& rayOrigin, 
        const Vector3f& rayDir,
        const Vector3f& v0, 
        const Vector3f& v1, 
        const Vector3f& v2,
        float& t)
    {
        const float EPSILON = 1e-6f;
        
        Vector3f edge1 = v1 - v0;
        Vector3f edge2 = v2 - v0;
        Vector3f h = rayDir.cross(edge2);
        float a = edge1.dot(h);
        
        if (a > -EPSILON && a < EPSILON)
            return false; // Ray is parallel to triangle
            
        float f = 1.0f / a;
        Vector3f s = rayOrigin - v0;
        float u = f * s.dot(h);
        
        if (u < 0.0f || u > 1.0f)
            return false;
            
        Vector3f q = s.cross(edge1);
        float v = f * rayDir.dot(q);
        
        if (v < 0.0f || u + v > 1.0f)
            return false;
            
        // Compute intersection distance
        t = f * edge2.dot(q);
        
        // Check if intersection is in front of ray origin
        return t > EPSILON;
    }

    /**
     * AABB Intersection Test
     */
    bool IntersectAABB(const Vector3f& rayOrigin, const Vector3f& rayDir, const Vector3& minB, const Vector3& maxB)
    {
        float tmin = (minB[0] - rayOrigin[0]) / rayDir[0];
        float tmax = (maxB[0] - rayOrigin[0]) / rayDir[0];

        if (tmin > tmax) std::swap(tmin, tmax);

        float tymin = (minB[1] - rayOrigin[1]) / rayDir[1];
        float tymax = (maxB[1] - rayOrigin[1]) / rayDir[1];

        if (tymin > tymax) std::swap(tymin, tymax);

        if ((tmin > tymax) || (tymin > tmax))
            return false;

        if (tymin > tmin)
            tmin = tymin;

        if (tymax < tmax)
            tmax = tymax;

        float tzmin = (minB[2] - rayOrigin[2]) / rayDir[2];
        float tzmax = (maxB[2] - rayOrigin[2]) / rayDir[2];

        if (tzmin > tzmax) std::swap(tzmin, tzmax);

        if ((tmin > tzmax) || (tzmin > tmax))
            return false;

        return true;
    }

    /**
     * Check if a shadow ray from intersection point to light source hits any geometry
     * OPTIMIZED: Uses Model Space Ray Tracing + AABB Culling
     */
    bool IsPointInShadow(
        const Vector3f& intersectionPoint,
        const Vector3f& lightPosition,
        OpenGLTriangleMesh* excludeMesh = nullptr)
    {
        const float EPSILON = 1e-4f;
        
        // Test against all meshes in the scene
        for (auto* mesh_obj : mesh_object_array) {
            
            // Skip self-shadowing or specific exclusions
            if (mesh_obj == excludeMesh) continue;
            
            // Cast to LanternObject to check AABB
            LanternObject* lantern = dynamic_cast<LanternObject*>(mesh_obj);
            if (!lantern) continue; // Only checking lanterns for now as they are the main occluders

            // 1. Transform Ray to Model Space
            Matrix4f modelMatrix;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    modelMatrix(i, j) = mesh_obj->model_matrix[j][i];
                }
            }
            
            // Invert to get World->Model transform
            Matrix4f invModel = modelMatrix.inverse();
            
            // Transform World Points to Model Space
            Vector4f originHom = invModel * Vector4f(intersectionPoint[0], intersectionPoint[1], intersectionPoint[2], 1.0f);
            Vector4f lightHom = invModel * Vector4f(lightPosition[0], lightPosition[1], lightPosition[2], 1.0f);
            
            Vector3f rayOriginLocal(originHom[0], originHom[1], originHom[2]);
            Vector3f lightPosLocal(lightHom[0], lightHom[1], lightHom[2]);
            
            Vector3f rayDirLocal = lightPosLocal - rayOriginLocal;
            float lightDistLocal = rayDirLocal.norm();
            
            if (lightDistLocal < 1e-6f) continue;
            
            rayDirLocal.normalize();
            
            // Offset origin slightly to avoid self-intersection issues
            rayOriginLocal += rayDirLocal * EPSILON;

            // 2. AABB Culling in Model Space
            // If ray doesn't hit the bounding box, it definitely doesn't hit the mesh
            if (!IntersectAABB(rayOriginLocal, rayDirLocal, lantern->min_aabb, lantern->max_aabb)) {
                continue;
            }

            // 3. Detailed Triangle Test in Model Space
            const auto& vertices = mesh_obj->mesh.Vertices();
            const auto& elements = mesh_obj->mesh.Elements();
            
            for (const auto& tri : elements) {
                // Vertices are already in Model Space
                const Vector3& v0 = vertices[tri[0]];
                const Vector3& v1 = vertices[tri[1]];
                const Vector3& v2 = vertices[tri[2]];
                
                // Convert to Vector3f for intersect function
                Vector3f v0f(v0[0], v0[1], v0[2]);
                Vector3f v1f(v1[0], v1[1], v1[2]);
                Vector3f v2f(v2[0], v2[1], v2[2]);
                
                float t;
                if (RayTriangleIntersect(rayOriginLocal, rayDirLocal, v0f, v1f, v2f, t)) {
                    // Check if intersection is valid and closer than the light
                    if (t > 0 && t < lightDistLocal) {
                        return true; // Occluded
                    }
                }
            }
        }
        
        return false; // No occlusion found
    }

    /**
     * Get all lantern positions (light sources) in world space
     * Returns a vector of lantern positions
     */
    std::vector<Vector3f> GetLanternPositions()
    {
        std::vector<Vector3f> positions;
        
        if (!glow_system)
            return positions;
            
        // Get positions from glow system particles
        // Index 0 is hero lantern, indices 1+ are background lanterns
        const auto& particlePositions = glow_system->particles.X();
        
        for (size_t i = 0; i < particlePositions->size(); ++i) {
            const Vector3& pos = (*particlePositions)[i];
            positions.push_back(Vector3f(static_cast<float>(pos[0]), static_cast<float>(pos[1]), static_cast<float>(pos[2])));
        }
        
        return positions;
    }

    /**
     * Compute shadow factor for a point from all lantern light sources
     * Returns a value between 0.0 (fully shadowed) and 1.0 (fully lit)
     * This can be used to modulate lighting contribution
     */
    float ComputeShadowFactor(const Vector3f& intersectionPoint, OpenGLTriangleMesh* excludeMesh = nullptr)
    {
        std::vector<Vector3f> lanternPositions = GetLanternPositions();
        
        if (lanternPositions.empty())
            return 1.0f; 
            
        int litCount = 0;
        int totalLights = lanternPositions.size();
        
        // Check shadow from each lantern
        for (const auto& lightPos : lanternPositions) {
            if (!IsPointInShadow(intersectionPoint, lightPos, excludeMesh)) {
                litCount++;
            }
        }
        
        // Return fraction of lights that are not blocked
        return static_cast<float>(litCount) / static_cast<float>(totalLights);
    }

    /**
     * Compute individual shadow factors for each lantern
     * Returns a vector where each element is 1.0 if that lantern is visible, 0.0 if shadowed
     */
    std::vector<float> ComputeShadowFactorsPerLight(const Vector3f& intersectionPoint, OpenGLTriangleMesh* excludeMesh = nullptr)
    {
        std::vector<Vector3f> lanternPositions = GetLanternPositions();
        std::vector<float> shadowFactors;
        
        for (const auto& lightPos : lanternPositions) {
            float factor = IsPointInShadow(intersectionPoint, lightPos, excludeMesh) ? 0.0f : 1.0f;
            shadowFactors.push_back(factor);
        }
        
        return shadowFactors;
    }

    // Shadow computation settings
    bool enable_shadows = true;
    int shadow_update_interval = 15; // Update shadows every N frames (15 = every 15 frames for performance)
    int frame_count = 0;
    float current_shadow_factor = 1.0f; // Current shadow factor (persists between updates)
    
    bool debug_shadows = false; // Set to true to see shadow computation in console (disabled for performance)

    /**
     * Compute shadow factors for vertices of a mesh
     * Stores shadow factors that can be passed to shader
     */
    void ComputeMeshShadowFactors(OpenGLTriangleMesh* mesh_obj, std::vector<float>& shadowFactors)
    {
        shadowFactors.clear();
        
        if (!enable_shadows || !mesh_obj)
            return;
            
        // Get mesh vertices in world space
        Matrix4f modelMatrix;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                modelMatrix(i, j) = mesh_obj->model_matrix[j][i];
            }
        }
        
        const auto& vertices = mesh_obj->mesh.Vertices();
        shadowFactors.reserve(vertices.size());
        
        // Compute shadow factor for each vertex
        for (const auto& vtx : vertices) {
            Vector3f worldPos = TransformPoint(modelMatrix, vtx);
            float shadowFactor = ComputeShadowFactor(worldPos, mesh_obj);
            shadowFactors.push_back(shadowFactor);
        }
    }

    /**
     * Update shadow data and pass to shaders
     * Computes shadows for key points and passes lantern positions to shaders
     */
    void UpdateShadowData()
    {
        // Get all lantern positions
        std::vector<Vector3f> lanternPositions = GetLanternPositions();
        
        if (lanternPositions.empty())
            return;
            
        // Get the lantern shader
        auto lanternShader = OpenGLShaderLibrary::Get_Shader("lantern");
        if (!lanternShader)
            return;
            
        lanternShader->Set_Uniform("num_lanterns", (int)lanternPositions.size());
        
        for (size_t i = 0; i < lanternPositions.size() && i < 32; ++i) {
            std::string uniformName = "lantern_positions[" + std::to_string(i) + "]";
            lanternShader->Set_Uniform(uniformName, Vector3f(lanternPositions[i][0], lanternPositions[i][1], lanternPositions[i][2]));
        }
        
        if (enable_shadows && (frame_count % shadow_update_interval == 0)) {
            float totalShadowFactor = 0.0f;
            int sampleCount = 0;
            
            if (hero_lantern) {
                Matrix4f heroMatrix;
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        heroMatrix(i, j) = hero_lantern->model_matrix[j][i];
                    }
                }
                const auto& vertices = hero_lantern->mesh.Vertices();
                int sampleStep = std::max(1, (int)vertices.size() / 3); // Sample ~3 vertices
                for (size_t v = 0; v < vertices.size() && sampleCount < 3; v += sampleStep) {
                    Vector3f worldPos = TransformPoint(heroMatrix, vertices[v]);
                    float shadowFactor = ComputeShadowFactor(worldPos, hero_lantern);
                    totalShadowFactor += shadowFactor;
                    sampleCount++;
                }
            }
            
            for (size_t i = 0; i < background_lanterns.size() && i < 2; ++i) {
                auto* lantern = background_lanterns[i];
                Matrix4f lanternMatrix;
                for (int j = 0; j < 4; j++) {
                    for (int k = 0; k < 4; k++) {
                        lanternMatrix(j, k) = lantern->model_matrix[k][j];
                    }
                }
                const auto& vertices = lantern->mesh.Vertices();
                int sampleStep = std::max(1, (int)vertices.size() / 2); // Sample ~2 vertices
                for (size_t v = 0; v < vertices.size() && v < 5; v += sampleStep) {
                    Vector3f worldPos = TransformPoint(lanternMatrix, vertices[v]);
                    float shadowFactor = ComputeShadowFactor(worldPos, lantern);
                    totalShadowFactor += shadowFactor;
                    sampleCount++;
                }
            }
            
            if (sampleCount > 0) {
                current_shadow_factor = totalShadowFactor / static_cast<float>(sampleCount);
            } else {
                current_shadow_factor = 1.0f; 
            }
            
            if (debug_shadows && frame_count % 60 == 0) {
                std::cout << "Average shadow factor: " << current_shadow_factor 
                          << " (from " << sampleCount << " vertex samples)" << std::endl;
            }
        }
        
        // Always set shadow factor (even if not updated this frame, use previous value)
        lanternShader->Set_Uniform("global_shadow_factor", current_shadow_factor);
    }

    virtual void Toggle_Next_Frame()
    {
        float time = GLfloat(clock() - startTime) / CLOCKS_PER_SEC;
        
        OpenGLShaderLibrary::Get_Shader("lantern")->Set_Uniform("time", time);
        OpenGLShaderLibrary::Get_Shader("glow")->Set_Uniform("time", time);

        // Update trail shader time
        if (trail_system) {
            OpenGLShaderLibrary::Get_Shader("trail")->Set_Uniform("time", time);
        }

        if (water_plane) {
            OpenGLShaderLibrary::Get_Shader("water")->Set_Uniform("time", time);       
            water_plane->setTime(time);
        }

        for (auto &mesh_obj : mesh_object_array)
            mesh_obj->setTime(time);
        
        // Update shadow data (pass lantern positions to shader)
        UpdateShadowData();
        frame_count++;

        if (bgEffect){
            bgEffect->setResolution((float)Win_Width(), (float)Win_Height());
            bgEffect->setTime(time);
            bgEffect->setFrame(frame++);
        }

        if (skybox){
            skybox->setTime(time);
        }

        // --- Physics & Animation ---
        
        // Hero
        float hero_current_y = -5.0f;
        float hero_drift_x = 0.0f;
        if (hero_lantern) {
             float speed = 0.8f; // Slightly faster than background
             float y_start = -5.0f;
             hero_current_y = y_start + speed * time;
             
             // Complex drift for hero
             hero_drift_x = std::sin(time * 0.4f) * 0.8f + std::cos(time * 0.15f) * 0.5f;
             
             float swing_angle = std::sin(time * 1.0f) * 0.08f; 
             
             // Rotation
             float c = std::cos(swing_angle);
             float s = std::sin(swing_angle);
             Matrix4f rot;
             rot << c, -s, 0, 0,
                    s, c, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;

             // Translation
             Matrix4f trans;
             trans << 1, 0, 0, hero_drift_x,
                      0, 1, 0, hero_current_y,
                      0, 0, 1, 0,
                      0, 0, 0, 1;

             // Scale - Hero is huge now (2.5)
             Matrix4f scale;
             scale << 2.5, 0, 0, 0,
                      0, 2.5, 0, 0,
                      0, 0, 2.5, 0,
                      0, 0, 0, 1;

             Matrix4f t = trans * rot * scale;
             hero_lantern->Set_Model_Matrix(t);
             
             // Update Glow (Index 0)
             if(glow_system) {
                 (*glow_system->particles.X())[0] = Vector3(hero_drift_x, hero_current_y, 0);
             }
        }

        if (trail_system && hero_lantern) {
            float spawn_rate = 0.05f;
            
            if (time - last_trail_spawn_time > spawn_rate) {
                last_trail_spawn_time = time;
                
                TrailParticle trail;
                trail.index = next_trail_index;
                trail.birth_time = time;
                trail.velocity = Vector3((rand() % 100 - 50) / 200.0, -0.3, (rand() % 100 - 50) / 200.0);
                
                (*trail_system->particles.X())[trail.index] = Vector3(hero_drift_x, hero_current_y, 0);
                
                active_trails.push_back(trail);
                next_trail_index = (next_trail_index + 1) % 2000;
            }
            
            float lifetime = 8.0f;
            float dt = 0.016f;
            
            // Update all active trails
            for (auto it = active_trails.begin(); it != active_trails.end(); ) {
                float age = time - it->birth_time;
                
                if (age > lifetime) {
                    // Move particle far away and remove from active list
                    (*trail_system->particles.X())[it->index] = Vector3(10000, 10000, 10000);
                    it = active_trails.erase(it);
                } else {
                    // Update position - keep them moving
                    int idx = it->index;
                    Vector3 current_pos = (*trail_system->particles.X())[idx];
                    Vector3 new_pos(
                        current_pos[0] + it->velocity[0] * dt,
                        current_pos[1] + it->velocity[1] * dt,
                        current_pos[2] + it->velocity[2] * dt
                    );
                    (*trail_system->particles.X())[idx] = new_pos;
                    
                    ++it;
                }
            }
            
            trail_system->Set_Color(OpenGLColor(1.0f, 0.7f, 0.2f, 1.0f));
            trail_system->Set_Data_Refreshed();
        }
        // Background Lanterns
        for (size_t i = 0; i < background_lanterns.size(); ++i) {
             auto l = background_lanterns[i];
             
             std::mt19937 rng(42 + i); 
             std::uniform_real_distribution<float> dist_x(-40.0f, 40.0f);
             std::uniform_real_distribution<float> dist_y(-8.0f, 20.0f);
             std::uniform_real_distribution<float> dist_z(-30.0f, 30.0f);
             std::uniform_real_distribution<float> dist_scale(0.8f, 1.8f);
             
             float s_val = dist_scale(rng); 
             float x0 = dist_x(rng);
             float y0 = dist_y(rng);
             float z0 = dist_z(rng);
             if (std::abs(x0) < 2 && std::abs(y0 + 5) < 2) x0 += 5; 
             
             // Base movement - Slower rise, more complex drift
             // Rise speed ~0.4 (vs Hero 0.6)
             float current_base_y = y0 + 0.4f * time;
             
             // Exaggerated Turbulence
             float drift_y = std::sin(time * 0.5f + i) * 1.0f; 
             float drift_x = std::cos(time * 0.3f + i * 0.7f) * 2.0f + std::sin(time * 0.1f) * 1.5f;
             float drift_z = std::sin(time * 0.25f + i * 1.1f) * 1.5f;

             // Flocking Logic
             // As hero rises, lanterns get attracted to it
             float flock_strength = 0.0f;
             if (hero_lantern) {
                 // Strength increases but capped at 0.85 to prevent full collapse
                 flock_strength = (hero_current_y - (-5.0f)) / 25.0f; 
                 flock_strength = std::max(0.0f, std::min(flock_strength, 0.85f));
                 // Smooth easing
                 flock_strength = flock_strength * flock_strength * (3.0f - 2.0f * flock_strength);
             }

             // Calculate final position
             // Scatter position
             float x_scatter = x0 + drift_x;
             float y_scatter = current_base_y + drift_y;
             float z_scatter = z0 + drift_z;

             // Flock Target - LOOSE gathering
             // Target is Hero Position + 60% of original offset (maintains formation but tighter)
             float x_flock = hero_drift_x + (x0 * 0.6f); 
             float y_flock = hero_current_y + (y0 * 0.9f) - 3.0f; // Follow slightly below/around
             float z_flock = (z0 * 0.6f); 

             // Interpolate
             float final_x = x_scatter * (1.0f - flock_strength) + x_flock * flock_strength;
             float final_y = y_scatter * (1.0f - flock_strength) + y_flock * flock_strength;
             float final_z = z_scatter * (1.0f - flock_strength) + z_flock * flock_strength;

             float swing = std::sin(time * 1.5f + i) * 0.05f;
             float c = std::cos(swing);
             float s_sin = std::sin(swing);
             
             Matrix4f rot;
             rot << c, -s_sin, 0, 0,
                    s_sin, c, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;
                    
             Matrix4f trans;
             trans << 1, 0, 0, final_x,
                      0, 1, 0, final_y,
                      0, 0, 1, final_z,
                      0, 0, 0, 1;
                      
             Matrix4f scale;
             scale << s_val, 0, 0, 0,
                      0, s_val, 0, 0,
                      0, 0, s_val, 0,
                      0, 0, 0, 1;

             Matrix4f t = trans * rot * scale;
             l->Set_Model_Matrix(t);
             
             // Update Glow (Index 1 + i)
             if(glow_system) {
                 (*glow_system->particles.X())[1 + i] = Vector3(final_x, final_y, final_z);
             }
        }

        // Embers Simulation - Removed
        /*
        if (embers) {
            ...
        }
        */
        
        if(glow_system) glow_system->Set_Data_Refreshed();

        OpenGLViewer::Toggle_Next_Frame();
    }

    virtual void Run()
    {
        OpenGLViewer::Run();
    }
};

int main(int argc, char *argv[])
{
    MyDriver driver;
    driver.Initialize();
    driver.Run();
}

#endif