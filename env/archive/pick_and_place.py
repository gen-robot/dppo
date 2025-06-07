import cv2
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
import os
import sapien.core as sapien
import torch
import imageio
from collections import OrderedDict
from ..gym_utils.wrapper.base import BaseEnv, recover_action, get_pairwise_contact_impulse, get_pairwise_contacts

# from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat
from typing import List
from env.sapien_utils.math import wrap_to_pi, euler2quat, quat2euler, mat2euler, get_pose_from_rot_pos

from ..utils import apply_random_texture, check_intersect_2d, grasp_pose_process, check_intersect_2d_
from ..articulation.pick_and_place_articulation import (
    load_lab_wall,
    load_table_4,
    load_storage_box,
    build_actor_ycb,
    build_actor_egad,
    ASSET_DIR
)
from ..robot import load_robot_full
from ..controller.whole_body_controller import BaseArmSimpleController
import json
import pickle
import requests
from datetime import datetime

# from Projects.homebot.config import PANDA_DATA
PANDA_DATA = "/home/zhouzhiting/Data/panda_data"

class PickAndPlaceEnv(BaseEnv):
    def __init__(
            self,
            use_gui: bool,
            device: str,
            mipmap_levels=1,
            obs_keys=tuple(),
            action_relative="tool",
            door_from_urdf=False,
            domain_randomize=True,
            need_door_shut=True,
            use_real=False,
            canonical=True
    ):
        self.tcp_link_idx: int = None
        self.agv_link_idx: int = None
        self.door_handle_link_idx: int = None
        self.finger_link_idxs: List[int] = None
        self.observation_dict: dict = {}
        self.obs_keys = obs_keys
        self.action_relative = action_relative
        self.expert_phase = 0
        self.door_from_urdf = door_from_urdf
        self.domain_randomize = domain_randomize
        self.need_door_shut = need_door_shut
        self.use_real = use_real
        self.canonical = canonical
        super().__init__(use_gui, device, mipmap_levels)

        if self.use_real:
            for link in self.robot.get_links():
                if link.get_name() == "third_camera_link":
                    third_camera_mount_actor = link
                    break
            self.create_camera(
                None,
                None,
                None,
                "third",
                (320, 240),
                np.deg2rad(60),
                third_camera_mount_actor,
            )
        else:
            cam_p = np.array([-0.12, 0.3, 1.32])
            look_at_p = np.array([0.3, 0.05, 0.76])
            self.create_camera(
                position=cam_p,
                look_at_dir=look_at_p - cam_p,
                right_dir=np.array([0, 0, 1]),
                name="third",
                resolution=(320, 240),
                # resolution=(640, 480),
                fov=np.deg2rad(60),
            )

        # self.create_camera(
        #     position=np.array([-1, 1, 2]),
        #     # position=np.array([-1.7, -0.3, 1.0]),
        #     look_at_dir=np.array([1, 0, 1]) - np.array([-1, 1, 2]),
        #     right_dir=np.array([-1, -1, 0.0]),
        #     name="top",
        #     resolution=(640, 480),
        #     fov=np.deg2rad(60),
        # )
        self.standard_head_cam_pose = self.cameras["third"].get_pose()
        self.standard_head_cam_fovx = self.cameras["third"].fovx
        # camera_mount_actor = self.robot.get_links()[-1]
        camera_mount_actor = self.robot.get_links()[-2]
        # print(self.robot.get_links())
        # print(camera_mount_actor.name)
        # exit()

        self.create_camera(
            None, None, None, "wrist", (320, 240), np.deg2rad(60), camera_mount_actor
        )
        self.standard_wrist_cam_fovx = self.cameras["wrist"].fovx
        self.base_arm_controller = BaseArmSimpleController(self.robot)
        self.p_scale = 0.05 / 3
        self.rot_scale = 0.2 / 3
        self.vel_scale = np.array([1.0, 0.2])
        self.joint_scale = (
                                   -np.array(
                                       [
                                           -3.110177,
                                           -2.180000,
                                           -3.110177,
                                           -0.110000,
                                           -3.110177,
                                           -1.750000,
                                           -3.110177,
                                       ]
                                   )
                                   + np.array(
                               [3.110177, 2.180000, 3.110177, 3.110177, 3.110177, 3.110177, 3.110177]
                           )
                           ) / 2
        self.joint_mean = (
                                  np.array(
                                      [
                                          -3.110177,
                                          -2.180000,
                                          -3.110177,
                                          -0.110000,
                                          -3.110177,
                                          -1.750000,
                                          -3.110177,
                                      ]
                                  )
                                  + np.array(
                              [3.110177, 2.180000, 3.110177, 3.110177, 3.110177, 3.110177, 3.110177]
                          )
                          ) / 2
        # Set spaces
        ycb_models = json.load(open(os.path.join(ASSET_DIR, "mani_skill2_ycb", "info_pick_v3.json"), "r"))
        # self.model_db = ycb_models

        egad_models = json.load(open(os.path.join(ASSET_DIR, "mani_skill2_egad", "info_pick_train_v1.json"), "r"))
        # self.model_db = egad_models
        self.model_db = dict(
            ycb=ycb_models,
            egad=egad_models
        )

        self.reset(seed=0)
        self._update_observation()
        _obs_space_dict = {}
        for key in self.obs_keys:
            _obs_space_dict[key] = gym.spaces.Box(
                low=0 if "rgb" in key else -100,
                high=255 if "rgb" in key else 100,
                shape=self.observation_dict[key].shape,
                dtype=self.observation_dict[key].dtype,
            )
        self.observation_space = gym.spaces.Dict(_obs_space_dict)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )

    def load_scene(self):
        self.load_static()

        # self.robot = self.load_robot()
        self.robot, self.finger_link_idxs = load_robot_full(self.scene)
        if "free" in self.robot.get_name():
            qpos = self.robot.get_qpos()
            # NOTE: the rotation dofs are not the euler angles
            qpos[:6] = np.array([-0.5, 0.0, 0.5, np.pi, np.pi / 3, 0.0])
            self.robot.set_qpos(qpos)

        ###
        # self.blocks = load_blocks_on_table(self.scene)
        self.table_top_z = 0.76
        self.storage_box = load_storage_box(self.scene, root_position=np.array([0.2, -0.2, self.table_top_z]))
        # self.tables.append(self.storage_box)

        # self.objects = build_actor_ycb("011_banana", self.scene, root_position=np.array([0.3, 0.15, 1.0]))

    # def reload_objs(self, obj_list=None):
    #     if hasattr(self, "objs"):
    #         for obj_id, obj_dict in self.objs.items():
    #             self.scene.remove_actor(obj_dict["actor"])
    #         self.objs.clear()
    #     else:
    #         self.objs = dict()
    #
    #     if obj_list is None:
    #         obj_list = self.np_random.choice(list(self.model_db.keys()), 5, replace=False)
    #
    #     # obj_list = ["011_banana", ]
    #     for model_id in obj_list:
    #         bbox_min_z = self.model_db[model_id]["bbox"]["min"][-1] * self.model_db[model_id]["scales"][0]
    #         num_try = 0
    #         obj_invalid, init_p, init_angle, obj = True, None, None, None
    #         while num_try < 10 and obj_invalid:
    #             rand_p = self.np_random.uniform(-0.15, 0.15, size=(2,))
    #             init_p = np.array([rand_p[0], rand_p[1], 0]) + np.array(
    #                 [0.2, 0.2, self.table_top_z - bbox_min_z + 5e-3])
    #
    #             init_angle = self.np_random.uniform(-np.pi, np.pi)
    #
    #             obj = build_actor_ycb(
    #                 model_id, self.scene, root_position=init_p, root_angle=init_angle,
    #                 density=self.model_db[model_id]["density"],
    #                 scale=self.model_db[model_id]["scales"][0]
    #             )
    #             obj.set_damping(0.1, 0.1)
    #
    #             self.scene.step()
    #             all_contact = self.scene.get_contacts()
    #             # print(all_contact)
    #
    #             obj_invalid = False
    #             for prev_model_id, prev_model in self.objs.items():
    #                 # print(get_pairwise_contacts(all_contact, prev_model["actor"], obj))
    #                 if len(get_pairwise_contacts(all_contact, prev_model["actor"], obj)):
    #                     obj_invalid = True
    #                     self.scene.remove_actor(obj)
    #                     break
    #             num_try += 1
    #         print(model_id, init_p, init_angle)
    #
    #         if not obj_invalid:
    #             self.objs.update({model_id: dict(actor=obj, init_p=init_p, init_angle=init_angle)})
    #
    #     for model_id, obj_dict in self.objs.items():
    #         init_p = obj_dict["init_p"]
    #         init_angle = obj_dict["init_angle"]
    #         obj_dict["actor"].set_pose(
    #             sapien.Pose(
    #                 p=init_p,
    #                 q=np.array([np.cos(init_angle / 2), 0.0, 0.0, np.sin(init_angle / 2)])
    #             )
    #         )

    def reload_objs(self, obj_list=None, egad_ratio=0.5, num_obj=1):
        if hasattr(self, "objs"):
            for obj_id, obj_dict in self.objs.items():
                self.scene.remove_actor(obj_dict["actor"])
            self.objs.clear()
        else:
            self.objs = dict()

        if obj_list is None:
            num_egad = int(len(self.model_db["ycb"].keys()) / (1 - egad_ratio) * egad_ratio)
            num_egad = min(num_egad, len(self.model_db["egad"].keys()))
            egad_list = self.np_random.choice(list(self.model_db["egad"].keys()), num_egad, replace=False)
            egad_list = [("egad", model_id) for model_id in egad_list]
            ycb_list = [("ycb", model_id) for model_id in self.model_db["ycb"].keys()]
            obj_list = self.np_random.choice(egad_list + ycb_list, num_obj, replace=False)
            # print(obj_list)

        # obj_list = ["011_banana", ]
        for model_type, model_id in obj_list:
            bbox_min_z = self.model_db[model_type][model_id]["bbox"]["min"][-1] * \
                         self.model_db[model_type][model_id]["scales"][0]
            num_try = 0
            obj_invalid, init_p, init_angle, obj = True, None, None, None
            while num_try < 10 and obj_invalid:
                rand_p = self.np_random.uniform(-0.15, 0.15, size=(2,))
                init_p = np.array([rand_p[0], rand_p[1], 0]) + np.array(
                    [0.2, 0.2, self.table_top_z - bbox_min_z + 5e-3])

                init_angle = self.np_random.uniform(-np.pi, np.pi)

                obj_invalid = False
                if rand_p[0] > 0.1 and rand_p[1] > 0.1:
                    obj_invalid = True
                else:
                    for (prev_model_type, prev_model_id), prev_model in self.objs.items():
                        # print(get_pairwise_contacts(all_contact, prev_model["actor"], obj))
                        if check_intersect_2d_(init_p, self.model_db[model_type][model_id]["bbox"],
                                               self.model_db[model_type][model_id]["scales"][0],
                                               self.objs[(prev_model_type, prev_model_id)]["init_p"],
                                               self.model_db[prev_model_type][prev_model_id]["bbox"],
                                               self.model_db[prev_model_type][prev_model_id]["scales"][0]):
                            obj_invalid = True
                            break
                num_try += 1
            # print(model_id, init_p, init_angle)

            if not obj_invalid:
                if model_type == "egad":
                    mat = self.renderer.create_material()
                    color = self.np_random.uniform(0.2, 0.8, 3)
                    color = np.hstack([color, 1.0])
                    mat.set_base_color(color)
                    mat.metallic = 0.0
                    mat.roughness = 0.1
                    obj = build_actor_egad(
                        model_id, self.scene, root_position=init_p, root_angle=init_angle,
                        density=self.model_db[model_type][model_id]["density"] if "density" in
                                                                                  self.model_db[model_type][
                                                                                      model_id].keys() else 1000,
                        scale=self.model_db[model_type][model_id]["scales"][0],
                        render_material=mat,
                    )
                elif model_type == "ycb":
                    obj = build_actor_ycb(
                        model_id, self.scene, root_position=init_p, root_angle=init_angle,
                        density=self.model_db[model_type][model_id]["density"] if "density" in
                                                                                  self.model_db[model_type][
                                                                                      model_id].keys() else 1000,
                        scale=self.model_db[model_type][model_id]["scales"][0]
                    )
                    obj.set_damping(0.1, 0.1)
                else:
                    raise Exception("unknown data type!")

                self.objs.update({(model_type, model_id): dict(actor=obj, init_p=init_p, init_angle=init_angle)})

    # def load_objs(self):
    #     self.model_db = json.load(open(os.path.join(ASSET_DIR, "mani_skill2_ycb", "info_pick_v0.json"), "r"))
    #
    #     # obj_list = self.np_random.choice(list(self.model_db.keys()), 4, replace=False)
    #     obj_list = ["011_banana", ]
    #     self.objs = {}
    #     for model_id in obj_list:
    #         bbox_min_z = self.model_db[model_id]["bbox"]["min"][-1]
    #         num_try = 0
    #         p_invalid = True
    #         while num_try < 10 and p_invalid:
    #             rand_p = self.np_random.uniform(-0.15, 0.15, size=(2,))
    #             init_p = np.array([rand_p[0], rand_p[1], 0]) + np.array(
    #                 [0.3, 0.2, self.table_top_z - bbox_min_z + 1e-3])
    #             p_invalid = False
    #             for prev_model_id in self.objs.keys():
    #                 if check_intersect_2d(init_p, self.model_db[model_id]["bbox"],
    #                                       self.objs[prev_model_id][1], self.model_db[prev_model_id]["bbox"]):
    #                     p_invalid = True
    #                     break
    #             num_try += 1
    #         print(model_id, init_p)
    #         obj = build_actor_ycb(
    #             model_id, self.scene, root_position=init_p,
    #             density=self.model_db[model_id]["density"],
    #             scale=self.model_db[model_id]["scales"][0]
    #         )
    #         obj.set_damping(0.1, 0.1)
    #         self.objs.update({model_id: (obj, init_p)})

    def load_static(self):
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.light0 = self.scene.add_directional_light(
            np.array([1, -1, -1]), np.array([1.0, 1.0, 1.0]), shadow=True
        )
        # self.scene.add_directional_light([1, 0, -1], [0.9, 0.8, 0.8], shadow=False)
        self.scene.add_directional_light([0, 1, 1], [0.9, 0.8, 0.8], shadow=False)
        # self.scene.add_spot_light(
        #     np.array([0, 0, 1.5]),
        #     direction=np.array([0, 0, -1]),
        #     inner_fov=0.3,
        #     outer_fov=1.0,
        #     color=np.array([0.5, 0.5, 0.5]),
        #     shadow=False,
        # )

        physical_material = self.scene.create_physical_material(1.0, 1.0, 0.0)
        render_material = self.renderer.create_material()
        render_material.set_base_color(np.array([0.1, 0.1, 0.1, 1.0]))
        self.room_ground = self.scene.add_ground(
            0.0, material=physical_material, render_material=render_material
        )
        ###
        self.table4 = load_table_4(self.scene)
        # self.table2 = load_table_2(self.scene, root_position=np.array([1., 2., 0]))
        # Add room walls
        self.room_wall1 = load_lab_wall(self.scene, [5.0, 0.0], 10.0)
        self.room_wall2 = load_lab_wall(self.scene, [0.0, -5.0], 10.0, np.pi / 2)
        # self.room_wall3 = load_lab_wall(self.scene, [0.0, 5.0], 10.0, np.pi / 2)

        self.walls = [
            self.room_wall1, self.room_wall2,  # self.room_wall3,
        ]
        self.tables = [
            self.table4,  # self.table2,
        ]

    def reset(self, seed: int = None, options: dict = None, obj_list: List = None):
        super().reset(seed, options)
        self.canonical_random, _ = seeding.np_random(seed)

        # Randomize properties in the beginning of episode
        if self.domain_randomize:
            table_rand = self.np_random.uniform(0, 1, (2,))
            table_rand_size = np.array([1.2 * table_rand[0], 0.3 * table_rand[1], 0.]) + np.array([0.8, 0.9, 0.03])
            leg_pos_x = table_rand_size[0] / 2 - 0.1
            leg_pos_y = table_rand_size[1] / 2 - 0.1
            table_position = np.array([table_rand_size[0] / 2, 0, 0])
            if hasattr(self, "table4"):
                self.scene.remove_actor(self.table4)
            self.table4 = load_table_4(
                self.scene,
                surface_size=table_rand_size,
                leg_pos_x=leg_pos_x,
                leg_pos_y=leg_pos_y,
                root_position=table_position,
            )
            self.tables = [self.table4]

            wall_rand = self.np_random.uniform(0, 1, (2,))
            wall1_rand = table_rand_size[0] + 0.05 + wall_rand[0] * 3
            wall2_rand = -1.5 - wall_rand[1] * 3
            if hasattr(self, "room_wall1"):
                self.scene.remove_actor(self.room_wall1)
            if hasattr(self, "room_wall2"):
                self.scene.remove_actor(self.room_wall2)
            self.room_wall1 = load_lab_wall(self.scene, [wall1_rand, 0.0], 10.0)
            self.room_wall2 = load_lab_wall(self.scene, [0.0, wall2_rand], 10.0, np.pi / 2)
            self.walls = [
                self.room_wall1, self.room_wall2,  # self.room_wall3,
            ]

            if not self.canonical:
                self.scene.set_ambient_light(np.tile(self.canonical_random.uniform(0, 1, (1,)), 3))
                self.scene.remove_light(self.light0)
                self.light0 = self.scene.add_directional_light(
                    self.canonical_random.uniform(-1, 1, (3,)),
                    np.array([1.0, 1.0, 1.0]),
                    shadow=True,
                )
            # body friction
            # for link in self.door_articulation.get_links():
            #     print("link name", link.get_name())
            #     for cs in link.get_collision_shapes():
            #         friction = np.random.uniform(0.1, 1.0)
            #         phys_mtl = self.scene.create_physical_material(static_friction=friction, dynamic_friction=friction, restitution=0.1)
            #         cs.set_physical_material(phys_mtl)
            # joint property
            # for joint in self.door_articulation.get_active_joints():
            #     print("joint name", joint.get_name())
            #     joint_friction = np.random.uniform(0.05, 0.2)
            #     joint.set_friction(joint_friction)
        if self.domain_randomize:
            # Visual property randomize
            if not self.canonical:
                # ground
                # print(len(self.room_ground.get_visual_bodies()[0].get_render_shapes()[0].material.base_color))
                for rs in self.room_ground.get_visual_bodies()[0].get_render_shapes():
                    apply_random_texture(rs.material, self.canonical_random)

                # walls
                for wall in self.walls:
                    for rs in wall.get_visual_bodies()[0].get_render_shapes():
                        apply_random_texture(rs.material, self.canonical_random)

                # tables
                for table in self.tables:
                    table_leg_materials = []

                    for body in table.get_visual_bodies():
                        for rs in body.get_render_shapes():
                            if body.name in ["upper_surface", "bottom_surface"]:
                                apply_random_texture(rs.material, self.canonical_random)
                            else:
                                table_leg_materials.append(rs.material)
                    apply_random_texture(table_leg_materials, self.canonical_random)

                # storage box
                storage_box_materials = []
                for body in self.storage_box.get_visual_bodies():
                    for rs in body.get_render_shapes():
                        if body.name in ["upper_surface", "bottom_surface"]:
                            apply_random_texture(rs.material, self.canonical_random)
                        else:
                            storage_box_materials.append(rs.material)
                apply_random_texture(storage_box_materials, self.canonical_random)

                # robot
                for link_idx, link in enumerate(self.robot.get_links()):
                    vb_list = link.get_visual_bodies()
                    for vb in vb_list:
                        for rs in vb.get_render_shapes():
                            apply_random_texture(rs.material, self.canonical_random)
                            # material = rs.material
                            # # black gripper
                            # material.set_base_color(np.array([0.0, 0.0, 0.0, 1.0]))
                            # rs.set_material(material)
            else:
                # ground
                self.room_ground.get_visual_bodies()[0].get_render_shapes()[0].material.set_base_color(
                    np.array(
                        [0.2, 0.2, 0.2, 1.0]
                    )
                )

                # walls
                for wall in self.walls:
                    for rs in wall.get_visual_bodies()[0].get_render_shapes():
                        rs.material.set_base_color(
                            np.array(
                                [0.3, 0.3, 0.3, 1.0]
                            )
                        )

                # tables
                for table in self.tables:
                    table_leg_materials = []
                    for body in table.get_visual_bodies():
                        for rs in body.get_render_shapes():
                            if body.name in ["upper_surface", "bottom_surface"]:
                                rs.material.set_base_color(
                                    np.array(
                                        [0.6, 0.6, 0.6, 1.0]
                                    )
                                )
                            else:
                                rs.material.set_base_color(
                                    np.array(
                                        [0.6, 0.6, 0.6, 1.0]
                                    )
                                )

                # storage box
                for body in self.storage_box.get_visual_bodies():
                    for rs in body.get_render_shapes():
                        if body.name in ["upper_surface", "bottom_surface"]:
                            rs.material.set_base_color(
                                np.array(
                                    [0.4, 0.2, 0.0, 1.0]
                                )
                            )
                        else:
                            rs.material.set_base_color(
                                np.array(
                                    [0.6, 0.3, 0.0, 1.0]
                                )
                            )

                # # robot
                # for link_idx, link in enumerate(self.robot.get_links()):
                #     vb_list = link.get_visual_bodies()
                #     for vb in vb_list:
                #         for rs in vb.get_render_shapes():
                #             material = rs.material
                #             # black gripper
                #             material.set_base_color(np.array([0.0, 0.0, 0.0, 1.0]))
                #             rs.set_material(material)

            # for link in self.door_articulation.get_links():
            #     if link.get_name() != "door_board":
            #         continue
            #     render_body = link.get_visual_bodies()[0]
            #     for rs in render_body.get_render_shapes():
            #         apply_random_texture(rs.material, self.np_random)
        if self.domain_randomize:
            # Camera pose
            # print("Camera pose", self.cameras["third"].get_pose())
            # print("fov", self.cameras["third"].fovx, self.cameras["third"].fovy)
            # Randomize camera pose
            if self.use_real:
                pass
            else:
                pos_rand_range = (-0.1, 0.1)
                rot_rand_range = (-0.2, 0.2)
                self.cameras["third"].set_pose(
                    sapien.Pose(
                        self.standard_head_cam_pose.p
                        + self.np_random.uniform(*pos_rand_range, size=(3,)),
                        euler2quat(
                            quat2euler(self.standard_head_cam_pose.q)
                            + self.np_random.uniform(*rot_rand_range, size=(3,))
                        ),
                    )
                )
            self.cameras["third"].set_fovx(
                self.standard_head_cam_fovx + self.np_random.uniform(-0.1, 0.1),
                compute_y=True,
            )
            self.cameras["wrist"].set_fovx(
                self.standard_wrist_cam_fovx + self.np_random.uniform(-0.1, 0.1),
                compute_y=True,
            )

        # ###
        # video_filename = "test2"
        # video_writer = imageio.get_writer(
        #     f"{video_filename}.mp4",
        #     fps=40,
        #     format="FFMPEG",
        #     codec="h264",
        # )
        # for _ in range(200):
        #     self.scene.step()
        #     rgb_images = self.render_all()
        #     rgb_image = rgb_images["third-rgb"]
        #     video_writer.append_data(rgb_image)
        #     print(self.obj.get_pose())
        # video_writer.close()
        # exit()

        # print(self.robot.get_name())
        if "free" in self.robot.get_name():
            # Robot pose randomize
            float_xyz_rpy = self.np_random.uniform(
                low=np.array([-0.5, -0.2, 1.3, np.pi / 6 * 6, -np.pi / 3, -np.pi / 12]),
                high=np.array([-0.3, 0.2, 1.6, np.pi / 6 * 6, -np.pi / 6, np.pi / 12]),
            )
            new_dofs = self.robot.get_qpos().copy()
            new_dofs[:6] = self._tcp_pose_to_float_dof(
                sapien.Pose(
                    p=float_xyz_rpy[:3],
                    q=euler2quat(float_xyz_rpy[3:6]),
                )
            )
            self.robot.set_qpos(new_dofs)
        else:
            if self.use_real:
                pass
                # init_p = self.np_random.uniform(
                #     low=np.array([-0.6, 0.4, 0.0]), high=np.array([-0.5, 0.4, 0.0])
                # )
                # init_angle = self.np_random.uniform(low=-0.1, high=0.1)
            else:
                init_p = self.np_random.uniform(
                    low=np.array([-0.2, 0.0, 0.0]), high=np.array([-0.1, 0.0, 0.0])
                )
                # init_p = np.array([-0.5, 0.0, 0.0])
                init_angle = self.np_random.uniform(
                    low=-np.pi / 2 - 0.01, high=-np.pi / 2 + 0.01
                )

            # init_angle = 0.0
            self.robot.set_root_pose(
                sapien.Pose(
                    init_p,
                    np.array(
                        [np.cos(init_angle / 2), 0.0, 0.0, np.sin(init_angle / 2)]
                    ),
                )
            )
            new_dofs = self.robot.get_qpos().copy()
            new_dofs[self.base_arm_controller.mobile_base_joint_indices] = 0.0
            # desired_relative_ee_pose = sapien.Pose(np.array([0.0, 0.0, 0.62]), euler2quat(np.array([0., 0., np.pi / 2]))).transform(
            #     sapien.Pose(np.array([0.4, 0.0, 0.5]), euler2quat(np.array([np.pi, 0.0, 0.])))
            # )
            # desired_ee_pose = self._get_agv_pose().transform(desired_relative_ee_pose)
            # target_qpos = self.base_arm_controller.compute_q_target(
            #     np.zeros(2), desired_ee_pose, 0.0
            # )[self.base_arm_controller.arm_joint_indices]
            # print(target_qpos)
            # new_dofs[self.base_arm_controller.arm_joint_indices] = np.array(
            #     [
            #         -2.4678290e00,
            #         -9.1231304e-01,
            #         8.1998336e-01,
            #         1.8495449e00,
            #         3.4986243e00,
            #         -8.3336508e-01,
            #         0.0,
            #     ]
            # )
            if self.use_real:
                new_dofs[self.base_arm_controller.arm_joint_indices] = np.array(
                    [
                        -0.9698,
                        -0.5976,
                        -0.3690,
                        1.1773,
                        -0.2073,
                        1.7393,
                        0.2554,
                    ]
                ) + self.np_random.uniform(-0.1, 0.1, size=(7,))
            else:
                new_dofs[self.base_arm_controller.arm_joint_indices] = np.array(
                    [
                        4.5306436e-04,
                        -4.1584689e-02,
                        3.2796859e-04,
                        1.9370439e00,
                        2.9931009e-05,
                        1.9786177e00,
                        7.8774774e-06,
                    ]
                ) + self.np_random.uniform(-0.1, 0.1, size=(7,))
            new_dofs[self.base_arm_controller.finger_joint_indices] = 0.0
            self.robot.set_qpos(new_dofs)
            self.robot.set_qvel(np.zeros_like(new_dofs))
            self.robot.set_qacc(np.zeros_like(new_dofs))

        self.init_agv_pose = self._get_agv_pose()
        # print("In reset, init_agv_pose", self.init_agv_pose)
        # reset stage for expert policy
        self.expert_phase = 0
        self.reset_tcp_pose = self._get_tcp_pose()
        self.reload_objs(obj_list=obj_list)

        # TODO: get obs
        self._update_observation()
        obs = OrderedDict()
        for key in self.obs_keys:
            obs[key] = self.observation_dict[key]

        return obs, {}

    def step(self, action: np.ndarray):
        action = action.copy()
        info = {}
        if "free" in self.robot.get_name():
            assert len(action) == 7
            target_qpos = np.concatenate([action[:6], np.ones(6) * action[6]])
            target_qvel = np.zeros(12)
        else:
            if len(action) == 9:
                # TODO: IK for arm, velocity control for mobile base
                # action is desired base w, desired base v, delta eef pose, binary gripper action in the specified frame
                cur_ee_pose = self._get_tcp_pose()
                cur_relative_ee_pose = self.init_agv_pose.inv().transform(cur_ee_pose)
                # action relative to tool frame
                if self.action_relative == "tool":
                    # desired_relative_ee_pose = cur_relative_ee_pose.transform(
                    #     sapien.Pose(
                    #         p=self.p_scale * action[2:5],
                    #         q=euler2quat(self.rot_scale * action[5:8]),
                    #     )
                    # )
                    desired_relative_ee_pose = cur_relative_ee_pose.transform(
                        sapien.Pose(
                            p=action[2:5],
                            q=euler2quat(action[5:8]),
                        )
                    )
                # action relative to fixed frame
                elif self.action_relative == "base":
                    desired_relative_ee_pose = sapien.Pose(
                        cur_relative_ee_pose.p + self.p_scale * action[2:5],
                        euler2quat(
                            quat2euler(cur_relative_ee_pose.q)
                            + self.rot_scale * action[5:8]
                        ),
                    )
                # pose in initial agv frame
                elif self.action_relative == "none":
                    desired_relative_ee_pose = sapien.Pose(
                        p=action[2:5], q=euler2quat(action[5:8])
                    )
                    # dirty
                    # action[8] = -action[8]
                desired_ee_pose = self.init_agv_pose.transform(desired_relative_ee_pose)
                # target_qpos = self.base_arm_controller.compute_q_target(
                #     np.zeros(2), desired_ee_pose, (action[8] + 1) / 2 * 0.85
                # )
                target_qpos = self.base_arm_controller.compute_q_target(
                    np.zeros(2), desired_ee_pose, 0.85 - action[8]
                )
                info["desired_relative_pose"] = np.concatenate(
                    [desired_relative_ee_pose.p, desired_relative_ee_pose.q]
                )
            else:
                assert len(action) == 10
                # target_arm_q = action[2:9] * self.joint_scale + self.joint_mean
                target_arm_q = action[2:9]
                # target_gripper_q = (0.85 - (action[9] + 1) / 2 * 0.85) * np.ones(6)
                target_gripper_q = (0.85 - action[9]) * np.ones(6)
                target_qpos = np.concatenate(
                    [np.zeros(2), target_arm_q, target_gripper_q]
                )
            info["desired_joints"] = target_qpos[
                self.base_arm_controller.arm_joint_indices
            ]
            info["desired_gripper_width"] = (
                    0.85 - target_qpos[self.base_arm_controller.finger_joint_indices[0]]
            )
            target_qvel = np.concatenate(
                [action[:2] * self.vel_scale, np.zeros(self.robot.dof - 2)]
            )
            # print("target qpos", target_qpos, "get qpos", self.robot.get_qpos())
        # target_qvel = recover_action(action, self.velocity_limit)
        # target_qvel[6:] = 0
        self.robot.set_drive_target(target_qpos)
        self.robot.set_drive_velocity_target(target_qvel)
        self.robot.set_qf(
            self.robot.compute_passive_force(
                external=False, coriolis_and_centrifugal=False
            )
        )
        for i in range(self.frame_skip):
            self.scene.step()

        # TODO: obs, reward, info
        # self._update_observation()
        obs = OrderedDict()
        for key in self.obs_keys:
            obs[key] = self.observation_dict[key]
        is_success = None
        reward = None
        # is_success = self._is_success()
        # reward = (
        #         self._reward_door_angle()
        #         + self._reward_handle_angle()
        #         + self._reward_approach_handle()
        # )  # TODO
        # print("In step, init_agv_pose", self.init_agv_pose)
        info.update(
            {
                "is_success": is_success,
                "init_agv_pose": self.init_agv_pose,
            }
        )
        return obs, reward, False, False, info

    def expert_action(self, obj_id, goal_obj_pose, noise_scale=0.0):
        # phases: before pregrasp, to grasp, close gripper, rotate, pull open
        actor = self.objs[obj_id]["actor"]
        init_p = self.objs[obj_id]["init_p"]
        init_angle = self.objs[obj_id]["init_angle"]
        along = self.model_db[obj_id[0]][obj_id[1]]["along"]
        obj_z_max = self.model_db[obj_id[0]][obj_id[1]]["bbox"]["max"][-1] * \
                    self.model_db[obj_id[0]][obj_id[1]]["scales"][0]
        done = False

        init_q = np.array(
            [np.cos(init_angle / 2), 0.0, 0.0, np.sin(init_angle / 2)]  # (w, x, y, z)
        )
        # print(obj_z_max, obj_id)

        if along == "y":
            obj_T_grasp = sapien.Pose.from_transformation_matrix(
                np.array(
                    [
                        [0, -1, 0, 0],
                        [-1, 0, 0, 0],
                        [0, 0, -1, max(obj_z_max - 0.04, -obj_z_max)],
                        [0, 0, 0, 1],
                    ]
                )
            )
        else:
            obj_T_grasp = sapien.Pose.from_transformation_matrix(
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, max(obj_z_max - 0.04, -obj_z_max)],
                        [0, 0, 0, 1],
                    ]
                )
            )

        obj_pose = actor.get_pose()

        desired_grasp_pose: sapien.Pose = None
        desired_gripper_width = None

        def apply_noise_to_pose(pose):
            pose.set_p(
                pose.p + self.np_random.uniform(-0.01, 0.01, size=(3,)) * noise_scale
            )
            pose.set_q(
                euler2quat(
                    quat2euler(pose.q)
                    + self.np_random.uniform(-0.1, 0.1, size=(3,)) * noise_scale
                )
            )

        if self.expert_phase == 0:
            obj_T_pregrasp = sapien.Pose(
                p=np.array([0.0, 0.0, 0.06 + obj_z_max]), q=obj_T_grasp.q
            )

            desired_grasp_pose = obj_pose.transform(obj_T_pregrasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)
            # print(self.expert_phase, desired_grasp_pose)
            apply_noise_to_pose(desired_grasp_pose)

            # randomize gripper width in phase 0
            # desired_gripper_width = self.np_random.uniform(0, 0.85)

            desired_gripper_width = 0.85 + self.np_random.uniform(-0.02, 0.02) * noise_scale

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )
        elif self.expert_phase == 1:
            desired_grasp_pose = obj_pose.transform(obj_T_grasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            # print(desired_grasp_pose, "desired")
            # print(self.expert_phase, desired_grasp_pose)
            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = 0.85 + self.np_random.uniform(-0.02, 0.02) * noise_scale

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                # 0.85 + self.np_random.uniform(-0.02, 0.02) * noise_scale,
                desired_gripper_width,
            )
        elif self.expert_phase == 2:
            gripper_width = self._get_gripper_width()
            desired_grasp_pose = obj_pose.transform(obj_T_grasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = gripper_width - 0.2 + self.np_random.uniform(-0.02, 0.02) * noise_scale

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )
        elif self.expert_phase == 3:
            gripper_width = self._get_gripper_width()

            obj_T_postgrasp = sapien.Pose(
                p=np.array([0.0, 0.0, 0.1 + obj_z_max]), q=obj_T_grasp.q
            )
            obj_init_pose = sapien.Pose(
                p=init_p, q=init_q
            )

            desired_grasp_pose = obj_init_pose.transform(obj_T_postgrasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            # print(self.expert_phase, desired_grasp_pose)
            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = gripper_width - 0.2 + self.np_random.uniform(-0.02, 0.02) * noise_scale

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )

        elif self.expert_phase == 4:
            gripper_width = self._get_gripper_width()

            obj_T_pregoal = sapien.Pose(
                p=np.array([0.0, 0.0, 0.1 + obj_z_max * 2]), q=obj_T_grasp.q
            )

            desired_grasp_pose = goal_obj_pose.transform(obj_T_pregoal)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            # print(self.expert_phase, desired_grasp_pose)
            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = gripper_width - 0.1 + self.np_random.uniform(-0.02, 0.02) * noise_scale

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )

        elif self.expert_phase == 5:
            desired_grasp_pose = obj_pose.transform(obj_T_grasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = 0.85 + self.np_random.uniform(-0.02, 0.02) * noise_scale

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )

        elif self.expert_phase == 6:
            desired_grasp_pose = self.reset_tcp_pose
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = 0.85 + self.np_random.uniform(-0.02, 0.02) * noise_scale

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )
        else:
            raise NotImplementedError
        # TODO: error recovery
        tcp_pose = self._get_tcp_pose()
        # print(tcp_pose, "tcp")
        if self.expert_phase == 0:
            # print(tcp_pose.p, desired_grasp_pose.p)
            # print(tcp_pose.q, desired_grasp_pose.q)
            # print(np.linalg.norm(tcp_pose.p - desired_grasp_pose.p))
            # print(abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]))
            if (
                    np.linalg.norm(tcp_pose.p - desired_grasp_pose.p) < 0.01
                    and abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]) > 0.95
            ):
                self.expert_phase = 1
        elif self.expert_phase == 1:
            if (
                    np.linalg.norm(tcp_pose.p - desired_grasp_pose.p) < 0.01
                    and abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]) > 0.95
            ):
                self.expert_phase = 2
        elif self.expert_phase == 2:
            if self._is_grasp(actor=actor, both_finger=True):
                self.expert_phase = 3

        elif self.expert_phase == 3:
            # print(tcp_pose.p, desired_grasp_pose.p)
            # print(tcp_pose.q, desired_grasp_pose.q)
            # print(np.linalg.norm(tcp_pose.p - desired_grasp_pose.p))
            # print(abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]))
            if (
                    np.linalg.norm(tcp_pose.p - desired_grasp_pose.p) < 0.01
                    and abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]) > 0.95
            ):
                self.expert_phase = 4

        elif self.expert_phase == 4:
            if (
                    np.linalg.norm(tcp_pose.p - desired_grasp_pose.p) < 0.01
                    and abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]) > 0.95
            ):
                self.expert_phase = 5

        elif self.expert_phase == 5:
            # pass
            if not self._is_grasp(actor):  # lost grasp, need to regrasp
                self.expert_phase = 6
        elif self.expert_phase == 6:
            if (
                    np.linalg.norm(tcp_pose.p - desired_grasp_pose.p) < 0.01
                    and abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]) > 0.95
            ):
                self.expert_phase = 0
                done = True

        return action, done, {"desired_grasp_pose": desired_grasp_pose,
                              "desired_gripper_width": desired_gripper_width}

    # compute all the observations
    def _update_observation(self):
        self.observation_dict.clear()
        image_obs = self.capture_images_new()
        world_tcp_pose = self._get_tcp_pose()
        tcp_pose = self.init_agv_pose.inv().transform(self._get_tcp_pose())
        gripper_width = self._get_gripper_width()
        # door_states = self.door_articulation.get_qpos()
        # handle_pose = self._get_handle_pose()
        arm_joints = self.robot.get_qpos()[self.base_arm_controller.arm_joint_indices]
        self.observation_dict.update(image_obs)
        self.observation_dict["tcp_pose"] = np.concatenate([tcp_pose.p, tcp_pose.q])
        self.observation_dict["gripper_width"] = gripper_width
        self.observation_dict["robot_joints"] = arm_joints
        # self.observation_dict["door_states"] = door_states
        self.observation_dict["privileged_obs"] = np.concatenate(
            [
                world_tcp_pose.p,
                world_tcp_pose.q,
                [gripper_width],
                # handle_pose.p,
                # handle_pose.q,
                # door_states,
            ]
        )

    def get_observation(self, use_image=True):
        obs = dict()
        if use_image:
            image_obs = self.capture_images_new()
            obs.update(image_obs)

        world_tcp_pose = self._get_tcp_pose()
        tcp_pose = self.init_agv_pose.inv().transform(self._get_tcp_pose())
        gripper_width = self._get_gripper_width()
        arm_joints = self.robot.get_qpos()[self.base_arm_controller.arm_joint_indices]
        obs["tcp_pose"] = np.concatenate([tcp_pose.p, tcp_pose.q])
        obs["gripper_width"] = gripper_width
        obs["robot_joints"] = arm_joints
        obs["privileged_obs"] = np.concatenate(
            [
                world_tcp_pose.p,
                world_tcp_pose.q,
                [gripper_width],
            ]
        )
        return obs

    def _reward_door_angle(self):
        return 0.01 * (self.door_articulation.get_qpos()[0] - np.pi / 10) / (np.pi / 10)

    def _reward_handle_angle(self):
        return (
                0.01
                * min(self.door_articulation.get_qpos()[1] - np.pi / 5, 0.0)
                / (np.pi / 5)
        )

    def _reward_approach_handle(self):
        tcp_pose = self._get_tcp_pose()
        handle_pose = self._get_handle_pose()
        return -0.002 * np.linalg.norm(tcp_pose.p - handle_pose.p)

    def _is_success(self):
        return self.door_articulation.get_qpos()[0] > np.pi / 10

    def _get_tcp_pose(self) -> sapien.Pose:
        """
        return tcp pose in world frame
        """
        if self.tcp_link_idx is None:
            for link_idx, link in enumerate(self.robot.get_links()):
                if link.get_name() == "link_tcp":
                    self.tcp_link_idx = link_idx
        link = self.robot.get_links()[self.tcp_link_idx]
        return link.get_pose()

    def _get_gripper_width(self) -> float:
        qpos = self.robot.get_qpos()
        return 0.85 - qpos[-6]

    def _get_agv_pose(self) -> sapien.Pose:
        if self.agv_link_idx is None:
            for link_idx, link in enumerate(self.robot.get_links()):
                if link.get_name() == "mount_link_fixed":
                    self.agv_link_idx = link_idx
        link = self.robot.get_links()[self.agv_link_idx]
        return link.get_pose()

    def _get_handle_pose(self) -> sapien.Pose:
        if self.door_handle_link_idx is None:
            target_link_name = (
                "door_handle" if not self.door_from_urdf else "link_switch_and_lock"
            )
            for link_idx, link in enumerate(self.door_articulation.get_links()):
                if link.get_name() == target_link_name:
                    self.door_handle_link_idx = link_idx
        link = self.door_articulation.get_links()[self.door_handle_link_idx]
        cs_list = link.get_collision_shapes()
        if not self.door_from_urdf:
            assert len(cs_list) == 3
        else:
            assert len(cs_list) == 2
        # level_mat = link.get_collision_shapes()[1].get_physical_material()
        if not self.door_from_urdf:
            handle_pose = link.get_pose().transform(cs_list[1].get_local_pose())
        else:
            # change local coordinate
            handle_pose = (
                link.get_pose()
                .transform(cs_list[0].get_local_pose())
                .transform(
                    sapien.Pose.from_transformation_matrix(
                        np.array(
                            [[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
                        )
                    )
                )
            )
        return handle_pose

    def _get_handle_joint_pose(self) -> sapien.Pose:
        if not self.door_from_urdf:
            joint_pose = self.door_articulation.get_active_joints()[1].get_global_pose()
        else:
            joint_pose = (
                self.door_articulation.get_active_joints()[1]
                .get_global_pose()
                .transform(
                    sapien.Pose.from_transformation_matrix(
                        np.array(
                            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
                        )
                    )
                )
            )
        # print("handle joint global pose", joint_pose)
        # if self.door_handle_link_idx is None:
        #     for link_idx, link in enumerate(self.door_articulation.get_links()):
        #         if link.get_name() == "door_handle":
        #             self.door_handle_link_idx = link_idx
        # link = self.door_articulation.get_links()[self.door_handle_link_idx]
        # cs_list = link.get_collision_shapes()
        # assert len(cs_list) == 3
        # handle_connector_pose = link.get_pose().transform(cs_list[0].get_local_pose())
        # print("handle connector pose", handle_connector_pose)
        return joint_pose

    def _get_door_joint_pose(self):
        if not self.door_from_urdf:
            joint_pose = self.door_articulation.get_active_joints()[0].get_global_pose()
        else:
            joint_pose = (
                self.door_articulation.get_active_joints()[0]
                .get_global_pose()
                .transform(
                    sapien.Pose.from_transformation_matrix(
                        np.array(
                            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
                        )
                    )
                )
            )
        return joint_pose

    def _tcp_pose_to_float_dof(self, tcp_pose: sapien.Pose) -> np.ndarray:
        # Convert tcp pose to gripper base
        tcp_T_gripperbase = sapien.Pose(p=np.array([0.0, 0.0, -0.172]))
        gripperbase_pose = tcp_pose.transform(tcp_T_gripperbase)
        dof_xyz = gripperbase_pose.p
        dof_rot = quat2euler(gripperbase_pose.q, axes="rxyz")
        return np.concatenate([dof_xyz, dof_rot])

    def _desired_tcp_to_action(
            self,
            tcp_pose: sapien.Pose,
            gripper_width: float,
            base_linv: float = 0,
            base_angv: float = 0,
    ) -> np.ndarray:
        assert self.action_relative != "none"
        if "free" in self.robot.get_name():
            float_dof = self._tcp_pose_to_float_dof(tcp_pose)
            gripper_qpos = np.array([0.85 - gripper_width])
            return np.concatenate([float_dof, gripper_qpos])
        else:
            cur_tcp_pose = self._get_tcp_pose()
            cur_relative_tcp_pose = self.init_agv_pose.inv().transform(cur_tcp_pose)
            desired_relative_tcp_pose = self.init_agv_pose.inv().transform(tcp_pose)
            # print("get tcp pose", cur_tcp_pose, "desired tcp pose", tcp_pose)
            # relative to tool frame
            if self.action_relative == "tool":
                curtcp_T_desiredtcp = cur_relative_tcp_pose.inv().transform(
                    desired_relative_tcp_pose
                )
                delta_pos = (
                        np.clip(curtcp_T_desiredtcp.p / self.p_scale, -1.0, 1.0)
                        * self.p_scale
                )
                delta_euler = (
                        np.clip(
                            wrap_to_pi(quat2euler(curtcp_T_desiredtcp.q)) / self.rot_scale,
                            -1.0,
                            1.0,
                        )
                        * self.rot_scale
                )
            # relative to fixed frame
            else:
                delta_pos = (
                        np.clip(
                            (desired_relative_tcp_pose.p - cur_relative_tcp_pose.p)
                            / self.p_scale,
                            -1.0,
                            1.0,
                        )
                        * self.p_scale
                )
                delta_euler = (
                        np.clip(
                            wrap_to_pi(
                                quat2euler(desired_relative_tcp_pose.q)
                                - quat2euler(cur_relative_tcp_pose.q)
                            )
                            / self.rot_scale,
                            -1.0,
                            1.0,
                        )
                        * self.rot_scale
                )
            base_action = np.array([base_angv, base_linv]) / self.vel_scale
            return np.concatenate(
                [
                    base_action,
                    delta_pos,
                    delta_euler,
                    [gripper_width],  # [(0.85 - gripper_width) / 0.85 * 2 - 1],
                ]
            )

    def _is_grasp(self, actor, threshold: float = 1e-4, both_finger=False):
        all_contact = self.scene.get_contacts()
        robot_finger_links: List[sapien.LinkBase] = [
            self.robot.get_links()[i] for i in self.finger_link_idxs
        ]
        finger_impulses = [
            get_pairwise_contact_impulse(
                all_contact, robot_finger, actor, None, None
            )
            for robot_finger in robot_finger_links
        ]
        finger_transforms = [
            robot_finger.get_pose().to_transformation_matrix()
            for robot_finger in robot_finger_links
        ]
        left_project_impulse = np.dot(finger_impulses[0], finger_transforms[0][:3, 1])
        right_project_impulse = np.dot(finger_impulses[1], -finger_transforms[1][:3, 1])
        if both_finger:
            return (
                    left_project_impulse > threshold and right_project_impulse > threshold
            )
        else:
            return left_project_impulse > threshold or right_project_impulse > threshold

        # print(all_contact)
        # print(actor.get_collision_shapes())
        # exit()

        # door_handle_link = self.door_articulation.get_links()[self.door_handle_link_idx]
        # if not self.door_from_urdf:
        #     door_handler_cs = door_handle_link.get_collision_shapes()[1]
        # else:
        #     door_handler_cs = door_handle_link.get_collision_shapes()[0]
        # finger_impulses = [
        #     get_pairwise_contact_impulse(
        #         all_contact, robot_finger, door_handle_link, None, door_handler_cs
        #     )
        #     for robot_finger in robot_finger_links
        # ]


def collect_rand_and_cano_data():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # video_filename = "test3"
    # video_writer = imageio.get_writer(
    #     f"{video_filename}.mp4",
    #     fps=40,
    #     format="FFMPEG",
    #     codec="h264",
    # )

    rand_pick_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        # obs_keys=("wrist-rgb", "tcp_pose", "gripper_width"),
        obs_keys=(),
        door_from_urdf=False,
        # use_real=True,
        domain_randomize=True,
        canonical=False
    )

    cano_pick_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        # obs_keys=("wrist-rgb", "tcp_pose", "gripper_width"),
        obs_keys=(),
        door_from_urdf=False,
        # use_real=True,
        domain_randomize=True,
        canonical=True
    )

    # envs = [rand_pick_env, cano_pick_env]
    envs = {
        "rand": rand_pick_env,
        "cano": cano_pick_env,
    }
    cameras = ["third", "wrist"]

    # save_dir = "/root/data/sim2sim_3"
    save_dir = os.path.join(PANDA_DATA, "sim2sim_3")

    # save_dir = "try"
    num_seeds = 10000
    steps_per_obj = 400
    # num_seeds = 10
    num_vid = 10
    os.makedirs(save_dir, exist_ok=True)

    cnt_list = []

    from tqdm import tqdm

    for seed in tqdm(range(num_seeds)):
        for idx, (env_key, env) in enumerate(envs.items()):
            env.reset(seed=seed)

            # env.scene.set_timestep(0.01)
            # env.frame_skip = 5

            random_state = np.random.RandomState(seed=seed)

            if seed < num_vid:
                video_writer = {cam: imageio.get_writer(
                    f"seed_{seed}_env_{env_key}_cam_{cam}.mp4",
                    # fps=40,
                    fps=10,
                    format="FFMPEG",
                    codec="h264",
                ) for cam in cameras}

            # data = []

            model_id_list = list(env.objs.keys())
            # print(model_id_list)
            random_state.shuffle(model_id_list)

            frame_id = 0

            for model_id in model_id_list:
                goal_p_rand = random_state.uniform(-0.1, 0.1, size=(2,))
                goal_q_rand = random_state.uniform(-0.5, 0.5)

                prev_privileged_obs = None
                for step in range(steps_per_obj):
                    action, done, _ = env.expert_action(
                        noise_scale=0.2, obj_id=model_id,
                        goal_obj_pose=sapien.Pose(
                            p=np.concatenate([np.array([0.2, -0.2]) + goal_p_rand, [0.76]]),
                            q=euler2quat(np.array([0, 0, goal_q_rand]))
                        )
                    )
                    _, _, _, _, info = env.step(action)

                    # rgb_images = env.render_all()
                    if frame_id % 10 == 0:
                        world_tcp_pose = env._get_tcp_pose()
                        gripper_width = env._get_gripper_width()
                        privileged_obs = np.concatenate(
                            [
                                world_tcp_pose.p,
                                world_tcp_pose.q,
                                [gripper_width],
                            ]
                        )
                        if prev_privileged_obs is not None and np.all(
                                np.abs(privileged_obs - prev_privileged_obs) < 1e-4):
                            env.expert_phase = 0
                            break
                        prev_privileged_obs = privileged_obs

                        rgb_images = env.capture_images_new(cameras=cameras)
                        # data_item = {}
                        # for k, image in rgb_images.items():
                        #     if "third" in k:
                        #         data_item[k] = image

                        if seed < num_vid:
                            for cam in cameras:
                                video_writer[cam].append_data(rgb_images[f"{cam}-rgb"])
                            # if seed == 0 and idx == 0 and step == 0:
                            #     # print(data_item.keys())
                            #     imageio.imwrite(f'third-rgb-0.jpg', rgb_images["third-rgb"])

                        # data.append(data_item)
                        # data.append(rgb_images["third-rgb"])
                        save_path = os.path.join(save_dir, f"seed_{seed}")
                        os.makedirs(save_path, exist_ok=True)
                        for cam in cameras:
                            imageio.imwrite(os.path.join(save_path, f"env_{env_key}_cam_{cam}_step_{frame_id}.jpg"),
                                            rgb_images[f"{cam}-rgb"])
                    frame_id += 1

                    if done:
                        # print(f"{model_id} done! use step {step}.")
                        # video_writer.close()
                        # p = envs[0].objs[model_id]["actor"].get_pose().p
                        # if 0.1 < p[0] < 0.4 and -0.35 < p[1] < -0.05:
                        #     success_list.append(model_id)
                        # else:
                        #     print(model_id, p)
                        break
                    elif step == steps_per_obj - 1:
                        # print(f"{model_id} step used out. ")
                        env.expert_phase = 0
                        # pass

            # print(len(frames))
            if idx == 0:
                cnt_list.append(frame_id)
                # cnt_list.append(len(data))

            # pickle.dump(data, open(os.path.join(save_dir, f"seed_{seed}_env_{idx}.pkl"), "wb"))

            if seed < num_vid:
                for cam in cameras:
                    video_writer[cam].close()

        # exit()

    print(np.sum(cnt_list))
    pickle.dump(cnt_list, open(os.path.join(save_dir, f"cnt.pkl"), "wb"))


def collect_imitation_data():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cano_pick_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        # obs_keys=("wrist-rgb", "tcp_pose", "gripper_width"),
        obs_keys=(),
        door_from_urdf=False,
        # use_real=True,
        domain_randomize=True,
        canonical=True
    )

    env = cano_pick_env
    cameras = ["third", "wrist"]

    # save_dir = "/root/data/cano_policy_2"
    save_dir = "/root/data/cano_policy_2"

    # save_dir = "try"
    # num_seeds = 10000
    num_seeds = 1000
    num_vid = 10
    os.makedirs(save_dir, exist_ok=True)

    # cnt_list = []

    from tqdm import tqdm

    for seed in tqdm(range(num_seeds)):
        save_path = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(save_path, exist_ok=True)

        env.reset(seed=seed)

        # env.scene.set_timestep(0.01)
        # env.frame_skip = 5

        random_state = np.random.RandomState(seed=seed)

        # data = []

        model_id_list = list(env.objs.keys())
        # print(model_id_list)
        random_state.shuffle(model_id_list)

        success_list = []

        for ep_id, model_id in enumerate(model_id_list):

            if seed < num_vid:
                video_writer = {cam: imageio.get_writer(
                    f"tmp/seed_{seed}_ep_{ep_id}_cam_{cam}.mp4",
                    # fps=40,
                    fps=20,
                    format="FFMPEG",
                    codec="h264",
                ) for cam in cameras}

            frame_id = 0
            success = False

            ep_path = os.path.join(save_path, f"ep_{ep_id}")
            os.makedirs(ep_path, exist_ok=True)
            goal_p_rand = random_state.uniform(-0.1, 0.1, size=(2,))
            goal_q_rand = random_state.uniform(-0.5, 0.5)

            prev_privileged_obs = None
            for step in range(500):
                action, done, desired_dict = env.expert_action(
                    noise_scale=0.5, obj_id=model_id,
                    goal_obj_pose=sapien.Pose(
                        p=np.concatenate([np.array([0.2, -0.2]) + goal_p_rand, [0.76]]),
                        q=euler2quat(np.array([0, 0, goal_q_rand]))
                    )
                )
                _, _, _, _, info = env.step(action)

                # rgb_images = env.render_all()
                # rgb_images = env.capture_images_new(cameras=cameras)
                if step < 400:
                    if done:
                        p = env.objs[model_id]["actor"].get_pose().p
                        if 0.05 < p[0] < 0.35 and -0.35 < p[1] < -0.05:
                            success = True
                        break
                    obs = env.get_observation()
                    obs.update({"action": action})
                    obs.update(desired_dict)
                    if prev_privileged_obs is not None and np.all(
                            np.abs(obs["privileged_obs"] - prev_privileged_obs) < 1e-4):
                        env.expert_phase = 0
                        break
                    prev_privileged_obs = obs["privileged_obs"]
                    # print(obs.keys())
                    # data_item = {}
                    # for k, image in rgb_images.items():
                    #     if "third" in k:
                    #         data_item[k] = image

                    # if seed < num_vid:
                    #     for cam in cameras:
                    #         video_writer[cam].append_data(obs[f"{cam}-rgb"])
                    # if seed == 0 and idx == 0 and step == 0:
                    #     # print(data_item.keys())
                    #     imageio.imwrite(f'third-rgb-0.jpg', rgb_images["third-rgb"])

                    # data.append(data_item)
                    # data.append(rgb_images["third-rgb"])

                    # for cam in cameras:
                    #     imageio.imwrite(os.path.join(ep_path, f"cam_{cam}_step_{frame_id}.jpg"),
                    #                     obs[f"{cam}-rgb"])
                    for cam in cameras:
                        image = obs.pop(f"{cam}-rgb")
                        imageio.imwrite(os.path.join(ep_path, f"step_{frame_id}_cam_{cam}.jpg"), image)
                        if seed < num_vid:
                            video_writer[cam].append_data(image)

                    pickle.dump(obs, open(os.path.join(ep_path, f"step_{frame_id}.pkl"), "wb"))
                    frame_id += 1

                else:
                    if done:
                        break
                    env.expert_phase = 6

            if success:
                success_list.append((ep_id, "s", frame_id))
                # print(seed, ep_id, "s", frame_id)
            else:
                success_list.append((ep_id, "f", frame_id))
                # print(seed, ep_id, "f", frame_id)

                # if done:
                # print(f"{model_id} done! use step {step}.")
                # video_writer.close()
                # p = envs[0].objs[model_id]["actor"].get_pose().p
                # if 0.1 < p[0] < 0.4 and -0.35 < p[1] < -0.05:
                #     success_list.append(model_id)
                # else:
                #     print(model_id, p)
                # break
                # elif step == 399:
                # print(f"{model_id} step used out. ")
                # pass

            # print(len(frames))
            # if idx == 0:
            #     cnt_list.append(frame_id)
            # cnt_list.append(len(data))

            # pickle.dump(data, open(os.path.join(save_dir, f"seed_{seed}_env_{idx}.pkl"), "wb"))

            if seed < num_vid:
                for writer in video_writer.values():
                    writer.close()

        # print(success_list)
        pickle.dump(success_list, open(os.path.join(save_path, f"info.pkl"), "wb"))

        # exit()

    # print(np.sum(cnt_list))
    # pickle.dump(cnt_list, open(os.path.join(save_dir, f"cnt.pkl"), "wb"))

    # rgb_images = open_door_sim.render_all()
    # imageio.imwrite(f'third-rgb-{"c" if canonical else "r"}-0.jpg', rgb_images["third-rgb"])
    #
    # open_door_sim.reset(seed=0)
    # rgb_images = open_door_sim.render_all()
    # imageio.imwrite(f'third-rgb-{"c" if canonical else "r"}-1.jpg', rgb_images["third-rgb"])
    #
    # open_door_sim.reset(seed=1)
    # rgb_images = open_door_sim.render_all()
    # imageio.imwrite(f'third-rgb-{"c" if canonical else "r"}-2.jpg', rgb_images["third-rgb"])

    # open_door_sim.reset(seed=0, canonical=canonical)

    # print(open_door_sim.cameras)

    # rgb_images = open_door_sim.render_all()
    #
    # for k, image in rgb_images.items():
    #     # print(k)
    #     # print(image.shape, image.dtype, np.max(image), np.min(image))
    #     if "depth" in k:
    #         image = np.clip(image / np.max(image) * 255, 0, 255).astype(np.uint8)
    #         image = np.dstack([image] * 3)
    #     elif "segmentation" in k:
    #         image = np.clip(image / np.max(image) * 255, 0, 255).astype(np.uint8)
    #         image1 = np.dstack([image[..., 0]] * 3)
    #         image2 = np.dstack([image[..., 1]] * 3)
    #         image = np.dstack([image[..., 0], image[..., 0], image[..., 0]])
    #         imageio.imwrite(f'{k}-{"c" if canonical else "r"}-1.jpg', image1)
    #         imageio.imwrite(f'{k}-{"c" if canonical else "r"}-2.jpg', image2)
    #
    #     imageio.imwrite(f'{k}-{"c" if canonical else "r"}.jpg', image)

    # success_list = json.load(open("success_model_id.json", "r"))

    # success_list = []
    # for model_id in list(envs[1].model_db.keys()):
    #     # if model_id in success_list:
    #     #     continue
    #     print(model_id, "...")
    #     video_filename = model_id
    #     video_writer = imageio.get_writer(
    #         f"{1}-{video_filename}.mp4",
    #         fps=40,
    #         format="FFMPEG",
    #         codec="h264",
    #     )
    #     obj_list = [video_filename]
    #     envs[1].reset(seed=2, obj_list=obj_list)
    #
    #     # envs[1].reset(seed=2, obj_list=obj_list)
    #     # for _ in range(200):
    #     #     open_door_sim.scene.step()
    #     #     print(_)
    #     #     all_contact = open_door_sim.scene.get_contacts()
    #     #     print(all_contact)
    #     #
    #     #     rgb_images = open_door_sim.render_all()
    #     #     rgb_image = rgb_images["third-rgb"]
    #     #     video_writer.append_data(rgb_image)
    #     #     # print(open_door_sim.obj.get_pose())
    #     # video_writer.close()
    #
    #     for i in range(1, 500):
    #         action, done = envs[1].expert_action(noise_scale=0.0, obj_id=list(envs[1].objs.keys())[0],
    #                                              goal_obj_pose=sapien.Pose(
    #                                                  p=np.array([0.2, -0.2, 0.76]), q=euler2quat(np.array([0, 0, 0])
    #                                                                                              )
    #                                              ))
    #         _, _, _, _, info = envs[1].step(action)
    #
    #         rgb_images = envs[1].render_all()
    #         video_writer.append_data(rgb_images["third-rgb"])
    #         # if i % 10 == 0:
    #         # imageio.imwrite(f'third-rgb-{i // 10}.jpg', rgb_images["third-rgb"])
    #         # imageio.imwrite(f'top-rgb-{i // 10}.jpg', rgb_images["top-rgb"])
    #         if done:
    #             video_writer.close()
    #             p = envs[0].objs[model_id]["actor"].get_pose().p
    #             if 0.1 < p[0] < 0.4 and -0.35 < p[1] < -0.05:
    #                 success_list.append(model_id)
    #             else:
    #                 print(model_id, p)
    #             break
    #
    #     exit()
    #
    # json.dump(success_list, open("success_model_id_2.json", "w"), indent=4)

    # exit(1)

    # step_count = 0
    # traj_count = 0
    # success_count = 0
    # done = False
    # while traj_count < 10:
    #     action = open_door_sim.expert_action(noise_scale=0.5)
    #     _, _, _, _, info = open_door_sim.step(action)
    #     step_count += 1
    #     rgb_image = open_door_sim.render()
    #
    #     video_writer.append_data(rgb_image)
    #     done = info["is_success"] or step_count >= 600
    #     if done:
    #         traj_count += 1
    #         step_count = 0
    #         success_count += info["is_success"]
    #         open_door_sim.reset()
    # print("success rate", success_count / traj_count)
    # video_writer.close()


def eval_imitation_with_goal():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        # obs_keys=("wrist-rgb", "tcp_pose", "gripper_width"),
        obs_keys=(),
        door_from_urdf=False,
        # use_real=True,
        domain_randomize=True,
        canonical=True
    )

    goal_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        # obs_keys=("wrist-rgb", "tcp_pose", "gripper_width"),
        obs_keys=(),
        door_from_urdf=False,
        # use_real=True,
        domain_randomize=True,
        canonical=True
    )

    cameras = ["third", "wrist"]
    usage = ["obs", "goal"]

    save_dir = "tmp"
    # save_dir = "try"
    # num_seeds = 10000
    num_eval = 10
    os.makedirs(save_dir, exist_ok=True)

    # cnt_list = []

    from tqdm import tqdm

    for i_eval in tqdm(range(num_eval)):
        seed = i_eval + 1000
        save_path = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(save_path, exist_ok=True)

        env.reset(seed=seed)
        goal_env.reset(seed=seed)

        random_state = np.random.RandomState(seed=seed)

        model_id_list = list(env.objs.keys())
        # print(model_id_list)
        random_state.shuffle(model_id_list)

        # success_list = []

        for ep_id, model_id in enumerate(model_id_list):

            frame_id = 0
            success = False

            ep_path = os.path.join(save_path, f"ep_{ep_id}")
            os.makedirs(ep_path, exist_ok=True)
            goal_p_rand = random_state.uniform(-0.1, 0.1, size=(2,))
            goal_q_rand = random_state.uniform(-0.5, 0.5)

            prev_privileged_obs = None
            prev_obs = None
            for step in range(500):
                action, done = goal_env.expert_action(
                    noise_scale=0.0, obj_id=model_id,
                    goal_obj_pose=sapien.Pose(
                        p=np.concatenate([np.array([0.2, -0.2]) + goal_p_rand, [0.76]]),
                        q=euler2quat(np.array([0, 0, goal_q_rand]))
                    )
                )
                _, _, _, _, info = goal_env.step(action)

                # rgb_images = env.render_all()
                # rgb_images = env.capture_images_new(cameras=cameras)
                if step < 400:
                    if done:
                        p = goal_env.objs[model_id]["actor"].get_pose().p
                        if 0.05 < p[0] < 0.35 and -0.35 < p[1] < -0.05:
                            success = True
                        break
                    obs = goal_env.get_observation()
                    if prev_privileged_obs is not None and np.all(
                            np.abs(obs["privileged_obs"] - prev_privileged_obs) < 1e-4):
                        goal_env.expert_phase = 0
                        break
                    prev_privileged_obs = obs["privileged_obs"]
                    prev_obs = obs
                    # print(obs.keys())
                    # data_item = {}
                    # for k, image in rgb_images.items():
                    #     if "third" in k:
                    #         data_item[k] = image

                    # for cam in cameras:
                    #     video_writer[cam].append_data(obs[f"{cam}-rgb"])
                    # if seed == 0 and idx == 0 and step == 0:
                    #     # print(data_item.keys())
                    #     imageio.imwrite(f'third-rgb-0.jpg', rgb_images["third-rgb"])

                    # data.append(data_item)
                    # data.append(rgb_images["third-rgb"])

                    # for cam in cameras:
                    #     imageio.imwrite(os.path.join(ep_path, f"cam_{cam}_step_{frame_id}.jpg"),
                    #                     obs[f"{cam}-rgb"])
                    # pickle.dump(obs, open(os.path.join(ep_path, f"step_{frame_id}.pkl"), "wb"))
                    frame_id += 1

                else:
                    if done:
                        break
                    goal_env.expert_phase = 6

            if success:
                # success_list.append((ep_id, "s", frame_id))
                print(seed, ep_id, "s", frame_id)
                video_writer = {cam: imageio.get_writer(
                    os.path.join(ep_path, f"seed_{seed}_ep_{ep_id}_cam_{cam}.mp4"),
                    # fps=40,
                    fps=20,
                    format="FFMPEG",
                    codec="h264",
                ) for cam in cameras}

                for cam in cameras:
                    imageio.imwrite(os.path.join(ep_path, f"goal-{cam}.jpg"), prev_obs[f"{cam}-rgb"])

                for step in range(1000):
                    obs = env.get_observation()
                    for cam in cameras:
                        imageio.imwrite(os.path.join(ep_path, f"obs-{cam}.jpg"), obs[f"{cam}-rgb"])
                        video_writer[cam].append_data(obs[f"{cam}-rgb"])

                    files = {
                        f"{usg}-{cam}": open(os.path.join(ep_path, f"{usg}-{cam}.jpg"), "rb")
                        for cam in cameras for usg in usage
                    }
                    response = requests.post("http://localhost:9977/diffusion", files=files)
                    response = response.json()

                    action = np.array(response["action"])
                    delta_pos = action[:3]
                    gripper_width = action[-1]

                    mat_6 = action[3:9].reshape(3, 2)
                    mat_6[:, 0] = mat_6[:, 0] / np.linalg.norm(mat_6[:, 0])
                    mat_6[:, 1] = mat_6[:, 1] / np.linalg.norm(mat_6[:, 1])
                    z_vec = np.cross(mat_6[:, 0], mat_6[:, 1])
                    mat = np.c_[mat_6, z_vec]
                    assert mat.shape == (3, 3)

                    delta_euler = (
                            np.clip(
                                wrap_to_pi(quat2euler(mat2quat(mat))) / env.rot_scale,
                                -1.0,
                                1.0,
                            )
                            * env.rot_scale
                    )

                    action = np.concatenate(
                        [
                            [0, 0],
                            delta_pos,
                            delta_euler,
                            [gripper_width],  # [(0.85 - gripper_width) / 0.85 * 2 - 1],
                        ]
                    )

                    _, _, _, _, info = env.step(action)

                for writer in video_writer.values():
                    writer.close()

                exit()

            else:
                # success_list.append((ep_id, "f", frame_id))
                print(seed, ep_id, "f", frame_id)

                # if done:
                # print(f"{model_id} done! use step {step}.")
                # video_writer.close()
                # p = envs[0].objs[model_id]["actor"].get_pose().p
                # if 0.1 < p[0] < 0.4 and -0.35 < p[1] < -0.05:
                #     success_list.append(model_id)
                # else:
                #     print(model_id, p)
                # break
                # elif step == 399:
                # print(f"{model_id} step used out. ")
                # pass

            # print(len(frames))
            # if idx == 0:
            #     cnt_list.append(frame_id)
            # cnt_list.append(len(data))

            # pickle.dump(data, open(os.path.join(save_dir, f"seed_{seed}_env_{idx}.pkl"), "wb"))

            # env = deepcopy(goal_env)

        # print(success_list)
        # pickle.dump(success_list, open(os.path.join(save_path, f"info.pkl"), "wb"))


def eval_imitation():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        # obs_keys=("wrist-rgb", "tcp_pose", "gripper_width"),
        obs_keys=(),
        door_from_urdf=False,
        # use_real=True,
        domain_randomize=True,
        canonical=True,
        action_relative="none"
    )

    cameras = ["third", "wrist"]
    usage = ["obs"]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    save_dir = os.path.join("tmp", stamp)
    # save_dir = "try"
    # num_seeds = 10000
    num_eval = 10
    os.makedirs(save_dir, exist_ok=True)

    # cnt_list = []

    from tqdm import tqdm

    for i_eval in range(num_eval):
        seed = i_eval + 1000
        save_path = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(save_path, exist_ok=True)

        env.reset(seed=seed)

        random_state = np.random.RandomState(seed=seed)

        model_id_list = list(env.objs.keys())
        # print(model_id_list)
        random_state.shuffle(model_id_list)

        for ep_id, model_id in enumerate(model_id_list):

            ep_path = os.path.join(save_path, f"ep_{ep_id}")
            os.makedirs(ep_path, exist_ok=True)

            video_writer = {cam: imageio.get_writer(
                os.path.join(ep_path, f"seed_{seed}_ep_{ep_id}_cam_{cam}.mp4"),
                fps=20,
                format="FFMPEG",
                codec="h264",
            ) for cam in cameras}

            for step in tqdm(range(500)):
                obs = env.get_observation()
                for cam in cameras:
                    imageio.imwrite(os.path.join(ep_path, f"obs-{cam}.jpg"), obs[f"{cam}-rgb"])
                    video_writer[cam].append_data(obs[f"{cam}-rgb"])

                files = {
                    f"{usg}-{cam}": open(os.path.join(ep_path, f"{usg}-{cam}.jpg"), "rb")
                    for cam in cameras for usg in usage
                }

                pose = obs["tcp_pose"]
                pose_p, pose_q = pose[:3], pose[3:]
                pose_mat = quat2mat(pose_q)

                pose_at_obs = get_pose_from_rot_pos(
                    pose_mat, pose_p
                )

                pose_mat_6 = pose_mat[:, :2].reshape(-1)
                proprio_state = np.concatenate(
                    [
                        pose_p,
                        pose_mat_6,
                        np.array([obs["gripper_width"]]),
                    ]
                )
                data = {"proprio_state": json.dumps(proprio_state.tolist())}

                response = requests.post("http://localhost:9977/diffusion", files=files, data=data)
                response = response.json()
                # print(response)

                actions = np.array(response["action"])
                converted_actions = []

                for i in range(10):
                    action = actions[i]
                    mat_6 = action[3:9].reshape(3, 2)
                    mat_6[:, 0] = mat_6[:, 0] / np.linalg.norm(mat_6[:, 0])
                    mat_6[:, 1] = mat_6[:, 1] / np.linalg.norm(mat_6[:, 1])
                    z_vec = np.cross(mat_6[:, 0], mat_6[:, 1])
                    mat = np.c_[mat_6, z_vec]
                    # assert mat.shape == (3, 3)

                    pos = action[:3]
                    gripper_width = action[-1]

                    init_to_desired_pose = pose_at_obs @ get_pose_from_rot_pos(
                        mat, pos
                    )
                    pose_action = np.concatenate(
                        [
                            [0, 0],
                            init_to_desired_pose[:3, 3],
                            mat2euler(init_to_desired_pose[:3, :3]),
                            [gripper_width]
                        ]
                    )
                    converted_actions.append(pose_action)

                    _, _, _, _, info = env.step(pose_action)

            for writer in video_writer.values():
                writer.close()

            # exit()


if __name__ == "__main__":
    # collect_rand_and_cano_data()
    collect_imitation_data()
    # eval_imitation()
    # test()
