import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)
import pybullet_data

from gibson.data.datasets import get_model_path
from gibson.core.physics.scene_abstract import Scene
import pybullet as p

################################################# ADD
def parse_mtl_file(mtl_file):
    materials = {}
    current_material = None

    with open(mtl_file, 'r') as file:
        # for line in file:
        #     if line.startswith('newmtl'):
        #         _, material_name = line.strip().split(' ', 1)
        #         current_material = material_name
        #         materials[current_material] = {}
        #     elif current_material is not None:
        #         if line.startswith('Kd'):
        #             _, r, g, b = line.strip().split(' ', 3)
        #             materials[current_material]['color'] = (float(r), float(g), float(b))
        #####################################################################
        # lines = file.readlines()
        # for line in lines:
        #     parts = line.strip().split()
        #     if len(parts) > 0:
        #         if parts[0] == 'newmtl':
        #             material_name = parts[1]
        #             current_material = material_name
        #             materials[current_material] = {}
        #         elif parts[0] == 'map_Kd':  # Diffuse texture map
        #             texture_file = parts[1]
        #             materials[current_material]['map_Kd'] = texture_file
        #####################################################################
        for line in file:
            if line.startswith('newmtl'):
                _, material_name = line.strip().split(' ')
                current_material = material_name
                materials[current_material] = {}
            elif current_material is not None:
                key_values = line.strip().split(' ')
                key = key_values[0]
                values = key_values[1:]
                materials[current_material][key] = values
    return materials

def parse_obj_file(obj_file):
    vertices = []
    faces = []
    materials = {}
    return vertices, faces, materials
################################################# ADD

class BuildingScene(Scene):
    def __init__(self, robot, model_id, gravity, timestep, frame_skip, env = None):
        Scene.__init__(self, gravity, timestep, frame_skip, env)
        
        # # contains cpp_world.clean_everything()
        # # stadium_pose = cpp_household.Pose()
        # # if self.zero_at_running_strip_start_line:
        # #    stadium_pose.set_xyz(27, 21, 0)  # see RUN_STARTLINE, RUN_RAD constants
        
        # filename = os.path.join(get_model_path(model_id), "mesh_z_up.obj")
        # #filename = os.path.join(get_model_path(model_id), "3d", "blender.obj")
        # #textureID = p.loadTexture(os.path.join(get_model_path(model_id), "3d", "rgb.mtl"))

        # if robot.model_type == "MJCF":
        #     MJCF_SCALING = robot.mjcf_scaling
        #     scaling = [1.0/MJCF_SCALING, 1.0/MJCF_SCALING, 0.6/MJCF_SCALING]
        # else:
        #     scaling  = [1, 1, 1]
        # magnified = [2, 2, 2]

        # collisionId = p.createCollisionShape(p.GEOM_MESH, fileName=filename, meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)


        # view_only_mesh = os.path.join(get_model_path(model_id), "mesh_view_only_z_up.obj")
        # if os.path.exists(view_only_mesh):
        #     visualId = p.createVisualShape(p.GEOM_MESH,
        #                                fileName=view_only_mesh,
        #                                meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        # else:
        #     visualId = -1

        # boundaryUid = p.createMultiBody(baseCollisionShapeIndex = collisionId, baseVisualShapeIndex = visualId)
        # p.changeDynamics(boundaryUid, -1, lateralFriction=1)
        # #print(p.getDynamicsInfo(boundaryUid, -1))
        # self.scene_obj_list = [(boundaryUid, -1)]       # baselink index -1
        

        # planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        # self.ground_plane_mjcf = p.loadMJCF(planeName)
        
        # if "z_offset" in self.env.config:
        #     z_offset = self.env.config["z_offset"]
        # else:
        #     z_offset = -10 #with hole filling, we don't need ground plane to be the same height as actual floors

        # p.resetBasePositionAndOrientation(self.ground_plane_mjcf[0], posObj = [0,0,z_offset], ornObj = [0,0,0,1])
        # p.changeVisualShape(boundaryUid, -1, rgbaColor=[168 / 255.0, 164 / 255.0, 92 / 255.0, 1.0],
        #                     specularColor=[0.5, 0.5, 0.5])
        
        ##########################################################################################
        
        ################################################# ADD

        obj_file = os.path.join(get_model_path(model_id), "mesh_z_up.obj")
        mtl_file = os.path.join(get_model_path(model_id), "mesh_z_up.mtl")
        filtered_obj_file = os.path.join(get_model_path(model_id), "cylinder_mesh_z_up.obj")

        vertices, faces, materials_obj = parse_obj_file(obj_file)
        materials_mtl = parse_mtl_file(mtl_file)
        
        if robot.model_type == "MJCF":
            MJCF_SCALING = robot.mjcf_scaling
            scaling = [1.0/MJCF_SCALING, 1.0/MJCF_SCALING, 0.6/MJCF_SCALING]
        else:
            scaling  = [1, 1, 1]
        magnified = [2, 2, 2]

        collisionId = p.createCollisionShape(p.GEOM_MESH, fileName=obj_file, meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        collisionId_f = p.createCollisionShape(p.GEOM_MESH, fileName=filtered_obj_file, meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)


        rendering = True

        view_only_mesh = os.path.join(get_model_path(model_id), "mesh_view_only_z_up.obj")
        if os.path.exists(view_only_mesh):
            visualId = p.createVisualShape(p.GEOM_MESH,
                                           fileName=view_only_mesh,
                                           meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        else:
            visualId = -1

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # disable tinyrenderer, software (CPU) renderer, we don't use it here
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

        boundaryUid = p.createMultiBody(baseCollisionShapeIndex = collisionId, baseVisualShapeIndex = visualId)
        boundaryUid_f = p.createMultiBody(baseCollisionShapeIndex = collisionId_f, baseVisualShapeIndex = visualId)
        p.changeDynamics(boundaryUid, -1, lateralFriction=1)
        # print(p.getDynamicsInfo(boundaryUid, -1))
        self.scene_obj_list = [(boundaryUid, -1)]       # baselink index -1

        planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.ground_plane_mjcf = p.loadMJCF(planeName)
        
        if "z_offset" in self.env.config:
            z_offset = self.env.config["z_offset"]
        else:
            z_offset = -10 #with hole filling, we don't need ground plane to be the same height as actual floors

        p.resetBasePositionAndOrientation(self.ground_plane_mjcf[0], posObj=[0, 0, z_offset], ornObj=[0, 0, 0, 1])

        # for material_name, material_info in materials_mtl.items():
        #     if 'map_Kd' in material_info:
        #         texture_file = material_info['map_Kd']
        #         texture_path = os.path.join(get_model_path(model_id), os.path.dirname(mtl_file), texture_file)
        #         print(texture_path)
        #     else:
        #         texture_path = None
        
        if rendering == False:
            visualId = p.createVisualShape(p.GEOM_MESH, fileName=obj_file, rgbaColor=[168 / 255.0, 164 / 255.0, 92 / 255.0, 1.0], specularColor=[0.4, 0.4, 0], meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
            visualId_f = p.createVisualShape(p.GEOM_MESH, fileName=filtered_obj_file, rgbaColor=[255 / 255.0, 0 / 255.0, 0 / 255.0, 1.0], specularColor=[0.4, 0.4, 0], meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)

            p.createMultiBody(baseCollisionShapeIndex=boundaryUid, baseVisualShapeIndex=visualId
                            #   baseMass=1,
                            #   baseInertialFramePosition=[0, 0, 0],
                            #   basePosition=[x_pos, y_pos, z_height],
                            #   useMaximalCoordinates=True
                             )
            p.createMultiBody(baseCollisionShapeIndex=boundaryUid_f, baseVisualShapeIndex=visualId_f
                            #   baseMass=1,
                            #   baseInertialFramePosition=[0, 0, 0],
                            #   basePosition=[x_pos, y_pos, z_height],
                            #   useMaximalCoordinates=True
                             )
        else:
            p.changeVisualShape(boundaryUid, -1, rgbaColor=[168 / 255.0, 164 / 255.0, 92 / 255.0, 1.0], specularColor=[0.5, 0.5, 0.5])
            p.changeVisualShape(boundaryUid_f, -1, rgbaColor=[255 / 255.0, 0 / 255.0, 0 / 255.0, 1.0], specularColor=[0.5, 0.5, 0.5])
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
       
    ################################################# ADD
        
    def episode_restart(self):
        Scene.episode_restart(self)



class SinglePlayerBuildingScene(BuildingScene):
    multiplayer = False
    def __init__(self, robot, model_id, gravity, timestep, frame_skip, env = None):
        BuildingScene.__init__(self, robot, model_id, gravity, timestep, frame_skip, env)



