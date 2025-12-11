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

/* Custom Lantern Object with AABB */
class LanternObject : public OpenGLTriangleMesh {
public:
    float seed = 0.0f;
    Vector3 min_aabb;
    Vector3 max_aabb;

    virtual void Set_Shader_Parameters() const override {
        if (shader_programs.size() > 0) {
            shader_programs[0]->Set_Uniform("seed", seed);
        }
    }
};

/* Main Driver Class for the Lantern Festival Scene */
class MyDriver : public OpenGLViewer
{
    std::vector<OpenGLTriangleMesh *> mesh_object_array;
    OpenGLBgEffect *bgEffect = nullptr;
    OpenGLSkybox *skybox = nullptr;
    OpenGLParticles<Particles<3>>* glow_system = nullptr; 
    OpenGLParticles<Particles<3>>* trail_system = nullptr;
    clock_t startTime;

    OpenGLTriangleMesh* water_plane = nullptr;

    LanternObject* hero_lantern = nullptr;
    std::vector<LanternObject*> background_lanterns;

    struct TrailParticle {
        int index;          
        float birth_time;   
        Vector3 velocity;   
    };

    std::vector<TrailParticle> active_trails;
    int next_trail_index = 0;
    float last_trail_spawn_time = 0.0f;

public:
    /* Initialize viewer settings */
    virtual void Initialize()
    {
        draw_axes = false;
        startTime = clock();
        OpenGLViewer::Initialize();
        
        if (opengl_window) {
            opengl_window->camera_distance = 15.0f;
            opengl_window->camera_target = Vector3f(0, 5, 0);
            opengl_window->rotation_matrix.setIdentity();
        }
    }

    /* Initialize all scene data, shaders, textures, and objects */
    virtual void Initialize_Data()
    {
        OpenGLShaderLibrary::Instance()->Add_Shader_From_File("shaders/basic.vert", "shaders/basic.frag", "basic");
        OpenGLShaderLibrary::Instance()->Add_Shader_From_File("shaders/lantern.vert", "shaders/lantern.frag", "lantern"); 
        OpenGLShaderLibrary::Instance()->Add_Shader_From_File("shaders/glow.vert", "shaders/glow.frag", "glow");
        OpenGLShaderLibrary::Instance()->Add_Shader_From_File("shaders/skybox.vert", "shaders/skybox.frag", "skybox");

        OpenGLShaderLibrary::Instance()->Add_Shader_From_File("shaders/water.vert", 
                                                       "shaders/water.frag", 
                                                       "water");

        OpenGLShaderLibrary::Instance()->Add_Shader_From_File("shaders/trail.vert", 
                                                        "shaders/trail.frag", 
                                                        "trail");
        
        OpenGLTextureLibrary::Instance()->Add_Texture_From_File("tex/bunny_color.jpg", "bunny_color");
        OpenGLTextureLibrary::Instance()->Add_Texture_From_File("tex/bunny_normal.png", "bunny_normal");
        OpenGLTextureLibrary::Instance()->Add_Texture_From_File("tex/lantern_color.png", "lantern_color");
        OpenGLTextureLibrary::Instance()->Add_Texture_From_File("tex/lantern_normal.png", "lantern_normal");
        OpenGLTextureLibrary::Instance()->Add_Texture_From_File("tex/buzz_color.png", "buzz_color"); 

        opengl_window->Add_Light(Vector3f(3, 1, 3), Vector3f(0.2, 0.2, 0.2), Vector3f(0.8, 0.5, 0.2), Vector3f(0.5, 0.5, 0.5)); 
        opengl_window->Add_Light(Vector3f(0, 0, -5), Vector3f(0.1, 0.1, 0.1), Vector3f(0.9, 0.9, 0.9), Vector3f(0.5, 0.5, 0.5));

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

        {
            glow_system = Add_Interactive_Object<OpenGLParticles<Particles<3>>>();
            glow_system->particles.Add_Elements(1 + 100);
            glow_system->Initialize();
            
            glow_system->opengl_points.Initialize();
            glow_system->opengl_points.shader_programs[0] = OpenGLShaderLibrary::Get_Shader("glow");
            glow_system->Set_Color(OpenGLColor(1.0f, 0.5f, 0.0f, 1.0f)); 
        }

        {
            int max_trail_particles = 2000;
            trail_system = Add_Interactive_Object<OpenGLParticles<Particles<3>>>();
            trail_system->particles.Add_Elements(max_trail_particles);
            trail_system->Initialize();
            
            trail_system->opengl_points.Initialize();
            trail_system->opengl_points.shader_programs[0] = OpenGLShaderLibrary::Get_Shader("trail");
            trail_system->Set_Color(OpenGLColor(1.0f, 0.5f, 0.0f, 0.0f)); 
        }

        {
            water_plane = Create_Water_Plane(50.0f, 50); 
            
            Matrix4f t;
            t << 1, 0, 0, 0,
                0, 1, 0, -10,
                0, 0, 1, 0,
                0, 0, 0, 1;
            water_plane->Set_Model_Matrix(t);
            
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

        {
            hero_lantern = Add_Lantern_Object("obj/JapaneseLamp.obj", 123.4f);
            Matrix4f t;
            t << 0.5, 0, 0, 0,
                 0, 0.5, 0, -5, 
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

            if (std::abs(x) < 2 && std::abs(y + 5) < 2) x += 5; 

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

        for (auto &mesh_obj : mesh_object_array){
            Set_Polygon_Mode(mesh_obj, PolygonMode::Fill);
            Set_Shading_Mode(mesh_obj, ShadingMode::TexAlpha);
            mesh_obj->Set_Data_Refreshed();
            mesh_obj->Initialize();
        }
        Toggle_Play();
    }

    /* Helper helper to create a lantern object */
    LanternObject *Add_Lantern_Object(std::string obj_file_name, float seed)
    {
        auto mesh_obj = Add_Interactive_Object<LanternObject>();
        Array<std::shared_ptr<TriangleMesh<3>>> meshes;
        Obj::Read_From_Obj_File_Discrete_Triangles(obj_file_name, meshes);

        mesh_obj->mesh = *meshes[0];
        mesh_obj->seed = seed;
        
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

    /* Creates the water plane geometry */
    OpenGLTriangleMesh* Create_Water_Plane(float size, int subdivisions)
    {
        auto mesh_obj = Add_Interactive_Object<OpenGLTriangleMesh>();
        
        std::vector<Vector3> vertices;
        std::vector<Vector3i> triangles;
        std::vector<Vector2> uvs;
        
        float step = size / subdivisions;
        
        for (int z = 0; z <= subdivisions; ++z) {
            for (int x = 0; x <= subdivisions; ++x) {
                float px = -size/2 + x * step;
                float pz = -size/2 + z * step;
                vertices.push_back(Vector3(px, 0, pz));
                
                uvs.push_back(Vector2(float(x)/subdivisions, float(z)/subdivisions));
            }
        }
        
        for (int z = 0; z < subdivisions; ++z) {
            for (int x = 0; x < subdivisions; ++x) {
                int i0 = z * (subdivisions + 1) + x;
                int i1 = i0 + 1;
                int i2 = i0 + (subdivisions + 1);
                int i3 = i2 + 1;
                
                triangles.push_back(Vector3i(i0, i2, i1));
                triangles.push_back(Vector3i(i1, i2, i3));
            }
        }
        
        mesh_obj->mesh.Vertices() = vertices;
        mesh_obj->mesh.Elements() = triangles;
        
        std::vector<Vector3> normals(vertices.size(), Vector3(0, 1, 0));
        mesh_obj->mesh.Normals() = normals;
        
        mesh_obj->mesh.Uvs() = uvs;
 
        return mesh_obj;
    }

    /* Transform a point from model space to world space */
    Vector3f TransformPoint(const Matrix4f& transform, const Vector3& point)
    {
        Vector4f homogeneous(static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2]), 1.0f);
        Vector4f transformed = transform * homogeneous;
        return Vector3f(transformed[0], transformed[1], transformed[2]);
    }

    /* Computes ray intersection with a triangle */
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
            return false; 
            
        float f = 1.0f / a;
        Vector3f s = rayOrigin - v0;
        float u = f * s.dot(h);
        
        if (u < 0.0f || u > 1.0f)
            return false;
            
        Vector3f q = s.cross(edge1);
        float v = f * rayDir.dot(q);
        
        if (v < 0.0f || u + v > 1.0f)
            return false;
            
        t = f * edge2.dot(q);
        
        return t > EPSILON;
    }

    /* Checks intersection with Axis-Aligned Bounding Box */
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

    /* Checks if a point is in shadow by testing occlusion */
    bool IsPointInShadow(
        const Vector3f& intersectionPoint,
        const Vector3f& lightPosition,
        OpenGLTriangleMesh* excludeMesh = nullptr)
    {
        const float EPSILON = 1e-4f;
        
        for (auto* mesh_obj : mesh_object_array) {
            
            if (mesh_obj == excludeMesh) continue;
            
            LanternObject* lantern = dynamic_cast<LanternObject*>(mesh_obj);
            if (!lantern) continue; 

            Matrix4f modelMatrix;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    modelMatrix(i, j) = mesh_obj->model_matrix[j][i];
                }
            }
            
            Matrix4f invModel = modelMatrix.inverse();
            
            Vector4f originHom = invModel * Vector4f(intersectionPoint[0], intersectionPoint[1], intersectionPoint[2], 1.0f);
            Vector4f lightHom = invModel * Vector4f(lightPosition[0], lightPosition[1], lightPosition[2], 1.0f);
            
            Vector3f rayOriginLocal(originHom[0], originHom[1], originHom[2]);
            Vector3f lightPosLocal(lightHom[0], lightHom[1], lightHom[2]);
            
            Vector3f rayDirLocal = lightPosLocal - rayOriginLocal;
            float lightDistLocal = rayDirLocal.norm();
            
            if (lightDistLocal < 1e-6f) continue;
            
            rayDirLocal.normalize();
            
            rayOriginLocal += rayDirLocal * EPSILON;

            if (!IntersectAABB(rayOriginLocal, rayDirLocal, lantern->min_aabb, lantern->max_aabb)) {
                continue;
            }

            const auto& vertices = mesh_obj->mesh.Vertices();
            const auto& elements = mesh_obj->mesh.Elements();
            
            for (const auto& tri : elements) {
                const Vector3& v0 = vertices[tri[0]];
                const Vector3& v1 = vertices[tri[1]];
                const Vector3& v2 = vertices[tri[2]];
                
                Vector3f v0f(v0[0], v0[1], v0[2]);
                Vector3f v1f(v1[0], v1[1], v1[2]);
                Vector3f v2f(v2[0], v2[1], v2[2]);
                
                float t;
                if (RayTriangleIntersect(rayOriginLocal, rayDirLocal, v0f, v1f, v2f, t)) {
                    if (t > 0 && t < lightDistLocal) {
                        return true; 
                    }
                }
            }
        }
        
        return false; 
    }

    /* Returns a list of lantern positions in world space */
    std::vector<Vector3f> GetLanternPositions()
    {
        std::vector<Vector3f> positions;
        
        if (!glow_system)
            return positions;
            
        const auto& particlePositions = glow_system->particles.X();
        
        for (size_t i = 0; i < particlePositions->size(); ++i) {
            const Vector3& pos = (*particlePositions)[i];
            positions.push_back(Vector3f(static_cast<float>(pos[0]), static_cast<float>(pos[1]), static_cast<float>(pos[2])));
        }
        
        return positions;
    }

    /* Computes the shadow factor for a given point */
    float ComputeShadowFactor(const Vector3f& intersectionPoint, OpenGLTriangleMesh* excludeMesh = nullptr)
    {
        std::vector<Vector3f> lanternPositions = GetLanternPositions();
        
        if (lanternPositions.empty())
            return 1.0f; 
            
        int litCount = 0;
        int totalLights = lanternPositions.size();
        
        for (const auto& lightPos : lanternPositions) {
            if (!IsPointInShadow(intersectionPoint, lightPos, excludeMesh)) {
                litCount++;
            }
        }
        
        return static_cast<float>(litCount) / static_cast<float>(totalLights);
    }

    /* Calculates per-light shadow factors */
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

    bool enable_shadows = true;
    int shadow_update_interval = 15; 
    int frame_count = 0;
    float current_shadow_factor = 1.0f; 
    
    bool debug_shadows = false; 

    /* Computes shadow factors for the entire mesh */
    void ComputeMeshShadowFactors(OpenGLTriangleMesh* mesh_obj, std::vector<float>& shadowFactors)
    {
        shadowFactors.clear();
        
        if (!enable_shadows || !mesh_obj)
            return;
            
        Matrix4f modelMatrix;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                modelMatrix(i, j) = mesh_obj->model_matrix[j][i];
            }
        }
        
        const auto& vertices = mesh_obj->mesh.Vertices();
        shadowFactors.reserve(vertices.size());
        
        for (const auto& vtx : vertices) {
            Vector3f worldPos = TransformPoint(modelMatrix, vtx);
            float shadowFactor = ComputeShadowFactor(worldPos, mesh_obj);
            shadowFactors.push_back(shadowFactor);
        }
    }

    /* Updates global shadow data for shaders */
    void UpdateShadowData()
    {
        std::vector<Vector3f> lanternPositions = GetLanternPositions();
        
        if (lanternPositions.empty())
            return;
            
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
                int sampleStep = std::max(1, (int)vertices.size() / 3); 
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
                int sampleStep = std::max(1, (int)vertices.size() / 2); 
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
        
        lanternShader->Set_Uniform("global_shadow_factor", current_shadow_factor);
    }

    /* Simulation loop for each frame */
    virtual void Toggle_Next_Frame()
    {
        float time = GLfloat(clock() - startTime) / CLOCKS_PER_SEC;
        
        OpenGLShaderLibrary::Get_Shader("lantern")->Set_Uniform("time", time);
        OpenGLShaderLibrary::Get_Shader("glow")->Set_Uniform("time", time);

        if (trail_system) {
            OpenGLShaderLibrary::Get_Shader("trail")->Set_Uniform("time", time);
        }

        if (water_plane) {
            OpenGLShaderLibrary::Get_Shader("water")->Set_Uniform("time", time);       
            water_plane->setTime(time);
        }

        for (auto &mesh_obj : mesh_object_array)
            mesh_obj->setTime(time);
        
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

        float hero_current_y = -5.0f;
        float hero_drift_x = 0.0f;
        if (hero_lantern) {
             float speed = 0.8f; 
             float y_start = -5.0f;
             hero_current_y = y_start + speed * time;
             
             hero_drift_x = std::sin(time * 0.4f) * 0.8f + std::cos(time * 0.15f) * 0.5f;
             
             float swing_angle = std::sin(time * 1.0f) * 0.08f; 
             
             float c = std::cos(swing_angle);
             float s = std::sin(swing_angle);
             Matrix4f rot;
             rot << c, -s, 0, 0,
                    s, c, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;

             Matrix4f trans;
             trans << 1, 0, 0, hero_drift_x,
                      0, 1, 0, hero_current_y,
                      0, 0, 1, 0,
                      0, 0, 0, 1;

             Matrix4f scale;
             scale << 2.5, 0, 0, 0,
                      0, 2.5, 0, 0,
                      0, 0, 2.5, 0,
                      0, 0, 0, 1;

             Matrix4f t = trans * rot * scale;
             hero_lantern->Set_Model_Matrix(t);
             
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
            
            for (auto it = active_trails.begin(); it != active_trails.end(); ) {
                float age = time - it->birth_time;
                
                if (age > lifetime) {
                    (*trail_system->particles.X())[it->index] = Vector3(10000, 10000, 10000);
                    it = active_trails.erase(it);
                } else {
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
             
             float current_base_y = y0 + 0.4f * time;
             
             float drift_y = std::sin(time * 0.5f + i) * 1.0f; 
             float drift_x = std::cos(time * 0.3f + i * 0.7f) * 2.0f + std::sin(time * 0.1f) * 1.5f;
             float drift_z = std::sin(time * 0.25f + i * 1.1f) * 1.5f;

             float flock_strength = 0.0f;
             if (hero_lantern) {
                 flock_strength = (hero_current_y - (-5.0f)) / 25.0f; 
                 flock_strength = std::max(0.0f, std::min(flock_strength, 0.85f));
                 flock_strength = flock_strength * flock_strength * (3.0f - 2.0f * flock_strength);
             }

             float x_scatter = x0 + drift_x;
             float y_scatter = current_base_y + drift_y;
             float z_scatter = z0 + drift_z;

             float x_flock = hero_drift_x + (x0 * 0.6f); 
             float y_flock = hero_current_y + (y0 * 0.9f) - 3.0f; 
             float z_flock = (z0 * 0.6f); 

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
             
             if(glow_system) {
                 (*glow_system->particles.X())[1 + i] = Vector3(final_x, final_y, final_z);
             }
        }
        
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