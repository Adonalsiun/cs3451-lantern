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
    clock_t startTime;

    // Lanterns
    // Lanterns
    LanternObject* hero_lantern = nullptr;
    std::vector<LanternObject*> background_lanterns;

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
            glow_system->particles.Add_Elements(1 + 30);
            glow_system->Initialize();
            
            // Fix: Initialize points and Override shader
            glow_system->opengl_points.Initialize();
            glow_system->opengl_points.shader_programs[0] = OpenGLShaderLibrary::Get_Shader("glow");
            glow_system->Set_Color(OpenGLColor(1.0f, 0.5f, 0.0f, 1.0f)); // Orange glow
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
        int num_lanterns = 30;
        std::mt19937 rng(42); 
        std::uniform_real_distribution<float> dist_x(-15.0f, 15.0f);
        std::uniform_real_distribution<float> dist_y(-2.0f, 20.0f);
        std::uniform_real_distribution<float> dist_z(-10.0f, 10.0f);
        std::uniform_real_distribution<float> dist_scale(0.3f, 0.6f); 

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
        mesh_object_array.push_back(mesh_obj);
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
     * Check if a shadow ray from intersection point to light source hits any geometry
     * Returns true if the point is in shadow (ray is blocked)
     */
    bool IsPointInShadow(
        const Vector3f& intersectionPoint,
        const Vector3f& lightPosition,
        OpenGLTriangleMesh* excludeMesh = nullptr)
    {
        // Compute shadow ray direction and distance
        Vector3f rayDir = lightPosition - intersectionPoint;
        float lightDistance = rayDir.norm();
        
        if (lightDistance < 1e-6f)
            return false; 
            
        rayDir.normalize();
        
        Vector3f rayOrigin = intersectionPoint + rayDir * 1e-4f;
        
        // Test against all meshes in the scene
        for (auto* mesh_obj : mesh_object_array) {
            if (mesh_obj == excludeMesh)
                continue;
                
            // Get model matrix (transforms from model space to world space)
            Matrix4f modelMatrix;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    modelMatrix(i, j) = mesh_obj->model_matrix[j][i]; // GLM[col][row] -> Eigen(row,col)
                }
            }
            
            const auto& vertices = mesh_obj->mesh.Vertices();
            const auto& elements = mesh_obj->mesh.Elements();
            
            for (const auto& tri : elements) {
                // Get triangle vertices in model space (Vector3)
                const Vector3& v0_model = vertices[tri[0]];
                const Vector3& v1_model = vertices[tri[1]];
                const Vector3& v2_model = vertices[tri[2]];
                
                // Transform triangle vertices to world space
                Vector3f v0_world = TransformPoint(modelMatrix, v0_model);
                Vector3f v1_world = TransformPoint(modelMatrix, v1_model);
                Vector3f v2_world = TransformPoint(modelMatrix, v2_model);
                
                // Test ray-triangle intersection
                float t;
                if (RayTriangleIntersect(rayOrigin, rayDir, v0_world, v1_world, v2_world, t)) {
                    // Check if intersection is between point and light
                    if (t < lightDistance - 1e-4f) {
                        return true; // Ray is blocked, point is in shadow
                    }
                }
            }
        }
        
        return false; // No occlusion found, point is lit
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
             float speed = 1.0f;
             float y_start = -5.0f;
             hero_current_y = y_start + speed * time;
             
             hero_drift_x = std::sin(time * 0.5f) * 0.5f;
             float swing_angle = std::sin(time * 2.0f) * 0.05f; 
             
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
             
             // Scale
             Matrix4f scale;
             scale << 0.5, 0, 0, 0,
                      0, 0.5, 0, 0,
                      0, 0, 0.5, 0,
                      0, 0, 0, 1;

             Matrix4f t = trans * rot * scale;
             hero_lantern->Set_Model_Matrix(t);
             
             // Update Glow (Index 0)
             if(glow_system) {
                 (*glow_system->particles.X())[0] = Vector3(hero_drift_x, hero_current_y, 0);
             }
        }

        // Background Lanterns
        for (size_t i = 0; i < background_lanterns.size(); ++i) {
             auto l = background_lanterns[i];
             
             std::mt19937 rng(42 + i); 
             std::uniform_real_distribution<float> dist_x(-15.0f, 15.0f);
             std::uniform_real_distribution<float> dist_y(-2.0f, 20.0f);
             std::uniform_real_distribution<float> dist_z(-10.0f, 10.0f);
             std::uniform_real_distribution<float> dist_scale(0.5f, 1.2f);
             
             float s_val = dist_scale(rng); 
             float x0 = dist_x(rng);
             float y0 = dist_y(rng);
             float z0 = dist_z(rng);
             if (std::abs(x0) < 2 && std::abs(y0 + 5) < 2) x0 += 5; 
             
             float drift_y = std::sin(time * 0.3f + i) * 0.5f + (0.2f * time); 
             float drift_x = std::cos(time * 0.2f + i * 0.5f) * 0.3f;
             
             float swing = std::sin(time * 1.5f + i) * 0.05f;
             float c = std::cos(swing);
             float s_sin = std::sin(swing);
             
             Matrix4f rot;
             rot << c, -s_sin, 0, 0,
                    s_sin, c, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;
                    
             Matrix4f trans;
             trans << 1, 0, 0, x0 + drift_x,
                      0, 1, 0, y0 + drift_y,
                      0, 0, 1, z0,
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
                 (*glow_system->particles.X())[1 + i] = Vector3(x0 + drift_x, y0 + drift_y, z0);
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