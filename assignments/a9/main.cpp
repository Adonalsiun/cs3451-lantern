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

    virtual void Toggle_Next_Frame()
    {
        float time = GLfloat(clock() - startTime) / CLOCKS_PER_SEC;
        
        OpenGLShaderLibrary::Get_Shader("lantern")->Set_Uniform("time", time);
        OpenGLShaderLibrary::Get_Shader("glow")->Set_Uniform("time", time);

        for (auto &mesh_obj : mesh_object_array)
            mesh_obj->setTime(time);

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