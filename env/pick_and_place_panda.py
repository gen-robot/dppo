import numpy as np
import sapien.core as sapien
from sapien.core import Pose, LinkBase
import gymnasium as gym
from gymnasium.utils import seeding
from gym import spaces
from typing import Dict, Any, Tuple, List, Optional
import random
import math
import os
import time
from collections import OrderedDict
import imageio  # 添加imageio用于保存视频

from env.gym_utils.wrapper.base import BaseEnv, recover_action, get_pairwise_contact_impulse, get_pairwise_contacts

# from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat
from typing import List
from env.sapien_utils.math import wrap_to_pi, euler2quat, quat2euler, mat2euler, get_pose_from_rot_pos

from .utils import apply_random_texture, check_intersect_2d, grasp_pose_process, check_intersect_2d_
from .articulation.pick_and_place_articulation import (
    # load_lab_door,
    # generate_rand_door_config,
    load_lab_wall,
    # load_lab_scene_urdf,
    load_table_4,
    load_table_2,
    load_storage_box,
    load_blocks_on_table,
    build_actor_ycb,
    build_actor_egad,
    ASSET_DIR
)
from .robot import load_robot_panda
from .controller.whole_body_controller import ArmSimpleController
import json
import pickle
import requests
from datetime import datetime

class PickAndPlaceEnv(BaseEnv):
    def __init__(
            self,
            use_gui: bool,
            device: str,
            mipmap_levels=1,
            obs_keys=tuple(),
            action_relative="tool",
            domain_randomize=True,
            canonical=True,
            allow_dir=[],
            record_video=False,  # 添加视频录制参数
            video_dir="/home/susichang/sichang/dppo/videos/pick_and_place",  # 添加视频保存目录参数
            #max_episode_steps=200
    ):
        self.tcp_link_idx: int = None
        self.agv_link_idx: int = None
        self.door_handle_link_idx: int = None
        self.finger_link_idxs: List[int] = None
        self.observation_dict: dict = {}
        self.obs_keys = obs_keys
        self.action_relative = action_relative
        self.expert_phase = 0
        self.domain_randomize = domain_randomize
        self.canonical = canonical
        self.allow_dir = allow_dir
        self.max_episode_steps = 200
        self.current_step = 0
        
        # 视频录制相关参数
        self.record_video = record_video
        self.video_dir = video_dir
        self.video_frames = []
        if self.record_video:
            os.makedirs(self.video_dir, exist_ok=True)
        super().__init__(use_gui, device, mipmap_levels)

        cam_p = np.array([0.793, -0.056, 1.505])   
        look_at_dir = np.array([-0.616, 0.044, -0.787])  
        right_dir = np.array([0.036, 0.999, 0.027])
        self.create_camera(
            position=cam_p,
            look_at_dir=look_at_dir,
            right_dir=right_dir,
            name="third",
            resolution=(320, 240),
            fov=np.deg2rad(44),
        )
        # self.create_camera(
        #     position=np.array([0., 1., 1.2]),
        #     look_at_dir=look_at_p - np.array([0., 1., 1.2]),
        #     right_dir=np.array([-1, 0, 0]),
        #     name="forth",
        #     resolution=(320, 240),
        #     fov=np.deg2rad(80),
        # )

        self.standard_head_cam_pose = self.cameras["third"].get_pose()
        self.standard_head_cam_fovx = self.cameras["third"].fovx
        # camera_mount_actor = self.robot.get_links()[-2]
        # # print(self.robot.get_links())
        # # print(camera_mount_actor.name)
        # # exit()
        #
        # self.create_camera(
        #     None, None, None, "wrist", (320, 240), np.deg2rad(60), camera_mount_actor
        # )
        # self.standard_wrist_cam_fovx = self.cameras["wrist"].fovx
        self.arm_controller = ArmSimpleController(self.robot)
        self.p_scale = 0.01
        self.rot_scale = 0.04
        self.gripper_scale = 0.007
        self.gripper_limit = 0.04

        joint_low = np.array(
            [
                -2.9671,
                -1.8326,
                -2.9671,
                -3.1416,
                -2.9671,
                -0.0873,
                -2.9671,
            ]
        )
        joint_high = np.array(
            [
                2.9671,
                1.8326,
                2.9671,
                0.0,
                2.9671,
                3.8223,
                2.9671
            ]
        )
        self.reset_joint_values = np.array(
            # [0., -0.785, 0., -2.356, 0., 1.571, 0.785]
            [0., -0.85, 0., -2.8, 0., 2.1, 0.785]
        )
        self.joint_scale = (joint_high - joint_low) / 2
        self.joint_mean = (joint_high + joint_low) / 2
        # Set spaces
        # ycb_models = json.load(open(os.path.join(ASSET_DIR, "mani_skill2_ycb", "info_pick_v3.json"), "r"))
        ycb_models = json.load(open(os.path.join("asset", "mani_skill2_ycb", "info_pick_v0.json"), "r"))
        # self.model_db = ycb_models

        egad_models = json.load(open(os.path.join("asset", "mani_skill2_egad", "info_pick_train_v1.json"), "r"))
        # self.model_db = egad_models
        self.model_db = dict(
            ycb=ycb_models,
            egad=egad_models
        )

        self.reset(seed=0)
        self._update_observation()
        _obs_space_dict = {}
        for key in self.obs_keys:
            _obs_space_dict[key] = spaces.Box(
                low=0 if "rgb" in key else -100,
                high=255 if "rgb" in key else 100,
                shape=self.observation_dict[key].shape,
                dtype=self.observation_dict[key].dtype,
            )
        self.observation_space = spaces.Dict(_obs_space_dict)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )

    def load_scene(self):
        self.load_static()

        self.robot, self.finger_link_idxs = load_robot_panda(self.scene)

        self.table_top_z = 0.76
        self.storage_box = load_storage_box(self.scene, root_position=np.array([0.4, -0.2, self.table_top_z]))

    def reload_objs(self, obj_list=None, egad_ratio=0.5, num_obj=1):   # can change num_obj
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
            # egad_list +
            obj_list = self.np_random.choice(egad_list +ycb_list, num_obj, replace=False)
            # print(obj_list)

        # obj_list = ["011_banana", ]
        print("obj_list", obj_list)
        for model_type, model_id in obj_list:
            bbox_min_z = self.model_db[model_type][model_id]["bbox"]["min"][-1] * \
                         self.model_db[model_type][model_id]["scales"][0]
            num_try = 0
            #obj_allow_dir = self.model_db[model_type][model_id]["allow_dir"]
            obj_allow_dir = self.model_db[model_type][model_id].get("allow_dir", "along")
            obj_invalid, init_p, init_angle, obj = True, None, None, None
            while num_try < 10 and obj_invalid:
                rand_p = self.np_random.uniform(-0.15, 0.15, size=(2,))
                init_p = np.array([rand_p[0], rand_p[1], 0]) + np.array(
                    [0.4, 0.2, self.table_top_z - bbox_min_z + 5e-3])

                init_angle = self.np_random.uniform(-np.pi, np.pi)

                obj_invalid = False
                if rand_p[0] < -0.05 and rand_p[1] < -0.05:
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
                    if obj is None:
                        print(f"Warning: Failed to build EGAD actor for model {model_id}")
                        continue
                elif model_type == "ycb":
                    obj = build_actor_ycb(
                        model_id, self.scene, root_position=init_p, root_angle=init_angle,
                        density=self.model_db[model_type][model_id]["density"] if "density" in
                                                                                  self.model_db[model_type][
                                                                                      model_id].keys() else 1000,
                        scale=self.model_db[model_type][model_id]["scales"][0],
                        obj_allow_dir=obj_allow_dir
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
        self.room_wall1 = load_lab_wall(self.scene, [-1.0, 0.0], 10.0)
        # self.room_wall2 = load_lab_wall(self.scene, [0.0, -5.0], 10.0, np.pi / 2)

        self.walls = [
            self.room_wall1,  # self.room_wall2,  # self.room_wall3,
        ]
        self.tables = [
            self.table4,  # self.table2,
        ]

    def reset(self, seed: int = None, options: dict = None, obj_list: List = None):
        super().reset(seed, options)
        self.canonical_random, _ = seeding.np_random(seed)
        self.current_step = 0

        # Randomize properties in the beginning of episode
        if self.domain_randomize:
            table_rand = self.np_random.uniform(-1, 1, (2,))
            table_rand_size = np.array([0.1 * table_rand[0], 0.3 * table_rand[1], 0.]) + np.array([1.0, 1.8, 0.03])
            leg_pos_x = table_rand_size[0] / 2 - 0.1
            leg_pos_y = table_rand_size[1] / 2 - 0.1
            table_position = np.array([0.35, 0, 0])
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
            wall1_rand = table_position[0] - table_rand_size[0] / 2 - 0.3 - wall_rand[0] * 0.5
            # print(wall1_rand)
            # wall2_rand = -1.5 - wall_rand[1] * 3
            if hasattr(self, "room_wall1"):
                self.scene.remove_actor(self.room_wall1)
            # if hasattr(self, "room_wall2"):
            #     self.scene.remove_actor(self.room_wall2)
            self.room_wall1 = load_lab_wall(self.scene, [wall1_rand, 0.0], 10.0)
            # self.room_wall2 = load_lab_wall(self.scene, [0.0, wall2_rand], 10.0, np.pi / 2)
            self.walls = [
                self.room_wall1,  # self.room_wall2,  # self.room_wall3,
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
                        [0.3, 0.3, 0.3, 1.0]
                    )
                )

                # walls
                for wall in self.walls:
                    for rs in wall.get_visual_bodies()[0].get_render_shapes():
                        rs.material.set_base_color(
                            np.array(
                                [0.5, 0.5, 0.5, 1.0]
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
                                        # [0.6, 0.6, 0.6, 1.0]
                                        [0.1, 0.3, 0.1, 1.0]
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

                # for link_idx, link in enumerate(self.robot.get_links()):
                #     vb_list = link.get_visual_bodies()
                #     for vb in vb_list:
                #         for rs in vb.get_render_shapes():
                #             material = rs.material
                #             print(link, material.base_color)

        if self.domain_randomize:
            # Camera pose
            # print("Camera pose", self.cameras["third"].get_pose())
            # print("fov", self.cameras["third"].fovx, self.cameras["third"].fovy)
            # Randomize camera pose

            pos_rand_range = (-0.05, 0.05)
            rot_rand_range = (-0.1, 0.1)
            fov_rand_range = (-0.05, 0.05)
            self.cameras["third"].set_pose(
                Pose(
                    self.standard_head_cam_pose.p
                    + self.np_random.uniform(*pos_rand_range, size=(3,)),
                    euler2quat(
                        quat2euler(self.standard_head_cam_pose.q)
                        + self.np_random.uniform(*rot_rand_range, size=(3,))
                    ),
                )
            )
            self.cameras["third"].set_fovx(
                self.standard_head_cam_fovx + self.np_random.uniform(*fov_rand_range),
                compute_y=True,
            )
            # self.cameras["wrist"].set_fovx(
            #     self.standard_wrist_cam_fovx + self.np_random.uniform(-0.1, 0.1),
            #     compute_y=True,
            # )

        init_p = self.np_random.uniform(
            low=np.array([-0.01, -0.02, 0.78]), high=np.array([0.01, 0.02, 0.78])
        )

        init_angle = self.np_random.uniform(
            low=-0.01, high=0.01
        )

        # init_angle = 0.0
        self.robot.set_root_pose(
            Pose(
                init_p,
                np.array(
                    [np.cos(init_angle / 2), 0.0, 0.0, np.sin(init_angle / 2)]
                ),
            )
        )
        new_dofs = self.robot.get_qpos().copy()

        new_dofs[self.arm_controller.arm_joint_indices] = (self.reset_joint_values
                                                           + self.np_random.uniform(-0.1, 0.1, size=(7,)))
        new_dofs[self.arm_controller.finger_joint_indices] = self.gripper_limit
        self.robot.set_qpos(new_dofs)
        self.robot.set_qvel(np.zeros_like(new_dofs))
        self.robot.set_qacc(np.zeros_like(new_dofs))

        self.init_base_pose = self._get_base_pose()
        # print("In reset, init_base_pose", self.init_base_pose)
        # reset stage for expert policy
        self.expert_phase = 0
        self.reset_tcp_pose = self._get_tcp_pose()
        self.reload_objs(obj_list=obj_list)

        # TODO: get obs
        self._update_observation()
        obs = OrderedDict()
        for key in self.obs_keys:
            obs[key] = self.observation_dict[key]

        # 如果启用了视频录制，保存上一轮的视频并清空帧列表
        if self.record_video and len(self.video_frames) > 0:
            try:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                video_path = os.path.join(self.video_dir, f"episode_{timestamp}.mp4")
                print(f"正在保存视频到: {video_path}")
                imageio.mimsave(video_path, self.video_frames, fps=30)
                print(f"视频保存完成，共 {len(self.video_frames)} 帧")
                self.video_frames = []
            except Exception as e:
                print(f"保存视频时出错: {e}")

        return obs, {"terminated": False, "truncated": False}

    def compute_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        reward = 0.0
        
        # 获取当前TCP和目标物体的位置
        tcp_pose = self._get_tcp_pose()
        obj_pose = self.objs[list(self.objs.keys())[0]]["actor"].get_pose()
        
        # 计算TCP到物体的距离
        tcp_to_obj_dist = np.linalg.norm(obj_pose.p - tcp_pose.p)
        reaching_reward = 1 - np.tanh(2 * tcp_to_obj_dist)  # 降低缩放因子
        reward += reaching_reward
        
        # 检查是否抓取成功
        is_grasped = self._is_grasp(self.objs[list(self.objs.keys())[0]]["actor"], both_finger=True)
        if is_grasped:
            reward += 2.0  # 增加抓取奖励
        
        # 计算物体到目标位置的距离
        goal_pos = np.array([0.4, -0.2, self.table_top_z])  # 存储箱的位置
        obj_to_goal_dist = np.linalg.norm(goal_pos - obj_pose.p)
        place_reward = 1 - np.tanh(2 * obj_to_goal_dist)  # 降低缩放因子
        reward += place_reward
        
        # 计算静态奖励
        qvel = self.robot.get_qvel()
        qvel = qvel[:-2]  # 排除夹爪关节
        static_reward = 1 - np.tanh(2 * np.linalg.norm(qvel))  # 降低缩放因子
        reward += static_reward
        
        # 检查是否成功放置
        is_success = obj_to_goal_dist < 0.1 and is_grasped  # 放宽成功条件
        if is_success:
            reward = 10.0  # 增加成功奖励
            
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        return self.compute_dense_reward(obs, action, info) / 10.0  # 更新归一化因子

    def step(self, action: np.ndarray):
        action = action.copy()
        info = {}
        self.current_step += 1

        if len(action) == 7:
            # TODO: IK for arm, velocity control for mobile base
            # action is desired base w, desired base v, delta eef pose, binary gripper action in the specified frame
            cur_ee_pose = self._get_tcp_pose()
            cur_relative_ee_pose = self.init_base_pose.inv().transform(cur_ee_pose)
            # action relative to tool frame
            if self.action_relative == "tool":
                desired_relative_ee_pose = cur_relative_ee_pose.transform(
                    Pose(
                        p=action[:3],
                        q=euler2quat(action[3:6]),
                    )
                )
            # action relative to fixed frame
            elif self.action_relative == "base":
                desired_relative_ee_pose = Pose(
                    cur_relative_ee_pose.p + self.p_scale * action[:3],
                    euler2quat(
                        quat2euler(cur_relative_ee_pose.q)
                        + self.rot_scale * action[3:6]
                    ),
                )
                # pose in initial agv frame
            elif self.action_relative == "none":
                desired_relative_ee_pose = Pose(
                    p=action[:3], q=euler2quat(action[3:6])
                )
                # dirty
                # action[8] = -action[8]
            desired_ee_pose = self.init_base_pose.transform(desired_relative_ee_pose)

            target_qpos = self.arm_controller.compute_q_target(
                desired_ee_pose, action[6]
            )
            info["desired_relative_pose"] = np.concatenate(
                [desired_relative_ee_pose.p, desired_relative_ee_pose.q]
            )
        else:
            assert len(action) == 8
            target_arm_q = action[:7]
            target_gripper_q = action[8] * np.ones(2)
            target_qpos = np.concatenate(
                [target_arm_q, target_gripper_q]
            )
        info["desired_joints"] = target_qpos[
            self.arm_controller.arm_joint_indices
        ]
        info["desired_gripper_width"] = (
            target_qpos[self.arm_controller.finger_joint_indices[0]]
        )

        self.robot.set_drive_target(target_qpos)
        self.robot.set_qf(
            self.robot.compute_passive_force(
                external=False, coriolis_and_centrifugal=False
            )
        )
        for i in range(self.frame_skip):
            self.scene.step()
            
        # 如果启用了视频录制，捕获当前帧
        if self.record_video:
            try:
                # 使用正确的相机方法获取图像
                image = self.cameras["third"].get_color_rgba()
                # 将图像从RGBA转换为RGB
                image = image[..., :3]  # 只取RGB通道
                self.video_frames.append(image)
                if len(self.video_frames) % 10 == 0:  # 每10帧打印一次信息
                    print(f"已录制 {len(self.video_frames)} 帧")
            except Exception as e:
                print(f"录制视频时出错: {e}")

        self._update_observation()
        obs = OrderedDict()
        for key in self.obs_keys:
            obs[key] = self.observation_dict[key]
            
        # 计算奖励
        reward = self.compute_dense_reward(obs, action, info)
        
        # 检查是否成功
        obj_pose = self.objs[list(self.objs.keys())[0]]["actor"].get_pose()
        goal_pos = np.array([0.4, -0.2, self.table_top_z])
        obj_to_goal_dist = np.linalg.norm(goal_pos - obj_pose.p)
        is_success = obj_to_goal_dist < 0.1 and self._is_grasp(self.objs[list(self.objs.keys())[0]]["actor"], both_finger=True) 
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        info.update({
            "is_success": is_success,
            "success_rate": float(is_success),
            "init_base_pose": self.init_base_pose,
        })
        
        return obs, reward, is_success, terminated, truncated, info

    def _compute_reward(self):
        """计算奖励"""
        # 获取TCP和物体的位置
        tcp_pose = self._get_tcp_pose()
        obj_pose = self.objs[list(self.objs.keys())[0]]["actor"].get_pose()
        
        # 计算TCP到物体的距离
        tcp_to_obj_dist = np.linalg.norm(tcp_pose.p - obj_pose.p)
        
        # 基础奖励：负的距离奖励
        reward = -tcp_to_obj_dist
        
        # 如果TCP接近物体，给予额外奖励
        if tcp_to_obj_dist < 0.1:
            reward += 1.0
            
        # 如果成功抓取物体，给予更多奖励
        if self._is_grasping():
            reward += 2.0
            
        return reward
        
    def _is_terminated(self):
        """检查是否完成任务"""
        # 如果物体被成功抓取并移动到目标位置，则任务完成
        if self._is_grasping():
            obj_pose = self.objs[list(self.objs.keys())[0]]["actor"].get_pose()
            # 设置目标位置为存储箱的位置
            goal_pos = np.array([0.4, -0.2, self.table_top_z])
            obj_to_goal_dist = np.linalg.norm(obj_pose.p - goal_pos)
            return obj_to_goal_dist < 0.05
        return False
        
    def _is_grasping(self):
        """检查是否正在抓取物体"""
        gripper_width = self._get_gripper_width()
        return gripper_width < 0.04  # 夹爪闭合阈值

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

        # print(along)
        if along == "y":
            obj_T_grasp = Pose.from_transformation_matrix(
                np.array(
                    [
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, -1, max(obj_z_max - 0.04, -obj_z_max + 0.01)],
                        [0, 0, 0, 1],
                    ]
                )
            )
        else:
            obj_T_grasp = Pose.from_transformation_matrix(
                np.array(
                    [
                        [-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -1, max(obj_z_max - 0.04, -obj_z_max + 0.01)],
                        [0, 0, 0, 1],
                    ]
                )
            )

        obj_pose = actor.get_pose()
        # print("obj_pose", obj_pose, quat2euler(obj_pose.q))

        desired_grasp_pose: Pose = None
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
            obj_T_pregrasp = Pose(
                p=np.array([0.0, 0.0, 0.06 + obj_z_max]), q=obj_T_grasp.q
            )

            desired_grasp_pose = obj_pose.transform(obj_T_pregrasp)
            # print("desired_grasp_pose", desired_grasp_pose, quat2euler(desired_grasp_pose.q))
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)
            # print("processed desired_grasp_pose", desired_grasp_pose, quat2euler(desired_grasp_pose.q))

            # print(self.expert_phase, desired_grasp_pose)
            apply_noise_to_pose(desired_grasp_pose)

            # randomize gripper width in phase 0
            desired_gripper_width = self.np_random.uniform(0, self.gripper_limit)
            # desired_gripper_width = self.gripper_limit + self.np_random.uniform(-0.02, 0.02) * noise_scale

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )
            # print("action", action)
            # tcp_pose = self._get_tcp_pose()
            # print("tcp_pose", tcp_pose, quat2euler(tcp_pose.q))
            # dp = Pose(p=desired_grasp_pose.p, q=euler2quat(np.array([np.pi, 0, 0])))
            # oTd = obj_pose.inv().transform(dp)
            # print(oTd, quat2euler(oTd.q), get_pose_from_rot_pos(quat2mat(oTd.q), oTd.p))
            # exit()

        elif self.expert_phase == 1:
            desired_grasp_pose = obj_pose.transform(obj_T_grasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            # print(desired_grasp_pose, "desired")
            # print(self.expert_phase, desired_grasp_pose)
            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = (
                    self.gripper_limit
                    + self.np_random.uniform(-self.gripper_scale / 2, self.gripper_scale / 2) * noise_scale
            )
            desired_gripper_width = np.clip(desired_gripper_width, 0, self.gripper_limit)

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )
        elif self.expert_phase == 2:
            gripper_width = self._get_gripper_width()
            desired_grasp_pose = obj_pose.transform(obj_T_grasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = (
                    gripper_width - self.gripper_scale
                    + self.np_random.uniform(-self.gripper_scale / 2, self.gripper_scale / 2) * noise_scale
            )
            desired_gripper_width = np.clip(desired_gripper_width, 0, self.gripper_limit)

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )
        elif self.expert_phase == 3:
            gripper_width = self._get_gripper_width()

            obj_T_postgrasp = Pose(
                p=np.array([0.0, 0.0, 0.1 + obj_z_max]), q=obj_T_grasp.q
            )
            obj_init_pose = Pose(
                p=init_p, q=init_q
            )

            desired_grasp_pose = obj_init_pose.transform(obj_T_postgrasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            # print(self.expert_phase, desired_grasp_pose)
            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = (
                    gripper_width - self.gripper_scale
                    + self.np_random.uniform(-self.gripper_scale / 2, self.gripper_scale / 2) * noise_scale
            )
            desired_gripper_width = np.clip(desired_gripper_width, 0, self.gripper_limit)

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )

        elif self.expert_phase == 4:
            gripper_width = self._get_gripper_width()

            obj_T_pregoal = Pose(
                p=np.array([0.0, 0.0, 0.1 + obj_z_max * 2]), q=obj_T_grasp.q
            )

            desired_grasp_pose = goal_obj_pose.transform(obj_T_pregoal)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            # print(self.expert_phase, desired_grasp_pose)
            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = (
                    gripper_width - self.gripper_scale
                    + self.np_random.uniform(-self.gripper_scale / 2, self.gripper_scale / 2) * noise_scale
            )
            desired_gripper_width = np.clip(desired_gripper_width, 0, self.gripper_limit)

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )

        elif self.expert_phase == 5:
            desired_grasp_pose = obj_pose.transform(obj_T_grasp)
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = (
                    self.gripper_limit
                    + self.np_random.uniform(-self.gripper_scale / 2, self.gripper_scale / 2) * noise_scale
            )
            desired_gripper_width = np.clip(desired_gripper_width, 0, self.gripper_limit)

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )

        elif self.expert_phase == 6:
            desired_grasp_pose = self.reset_tcp_pose
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = (
                    self.gripper_limit
                    + self.np_random.uniform(-self.gripper_scale / 2, self.gripper_scale / 2) * noise_scale
            )
            desired_gripper_width = np.clip(desired_gripper_width, 0, self.gripper_limit)

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
        tcp_pose = self.init_base_pose.inv().transform(self._get_tcp_pose())
        gripper_width = self._get_gripper_width()
        arm_joints = self.robot.get_qpos()[self.arm_controller.arm_joint_indices]
        self.observation_dict.update(image_obs)
        self.observation_dict["tcp_pose"] = np.concatenate([tcp_pose.p, tcp_pose.q])
        self.observation_dict["gripper_width"] = gripper_width
        self.observation_dict["robot_joints"] = arm_joints
        # self.observation_dict["door_states"] = door_states
        self.observation_dict["privileged_obs"] = np.concatenate(
            [
                world_tcp_pose.p,
                world_tcp_pose.q,
                [gripper_width]
            ]
        )

    def get_observation(self, use_image=True):
        obs = dict()
        if use_image:
            image_obs = self.capture_images_new()
            obs.update(image_obs)

        world_tcp_pose = self._get_tcp_pose()
        tcp_pose = self.init_base_pose.inv().transform(self._get_tcp_pose())
        gripper_width = self._get_gripper_width()
        arm_joints = self.robot.get_qpos()[self.arm_controller.arm_joint_indices]
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

    def _get_tcp_pose(self) -> Pose:
        """
        return tcp pose in world frame
        """
        if self.tcp_link_idx is None:
            for link_idx, link in enumerate(self.robot.get_links()):
                if link.get_name() == "panda_grasptarget":
                    self.tcp_link_idx = link_idx
        link = self.robot.get_links()[self.tcp_link_idx]
        return link.get_pose()

    def _get_gripper_width(self) -> float:
        qpos = self.robot.get_qpos()
        return qpos[-2]

    def _get_base_pose(self) -> Pose:
        return self.robot.get_pose()
        # if self.agv_link_idx is None:
        #     for link_idx, link in enumerate(self.robot.get_links()):
        #         if link.get_name() == "mount_link_fixed":
        #             self.agv_link_idx = link_idx
        # link = self.robot.get_links()[self.agv_link_idx]
        # return link.get_pose()

    def _desired_tcp_to_action(
            self,
            tcp_pose: Pose,
            gripper_width: float,
    ) -> np.ndarray:
        assert self.action_relative != "none"

        cur_tcp_pose = self._get_tcp_pose()
        cur_relative_tcp_pose = self.init_base_pose.inv().transform(cur_tcp_pose)
        desired_relative_tcp_pose = self.init_base_pose.inv().transform(tcp_pose)
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
        return np.concatenate(
            [
                delta_pos,
                delta_euler,
                [gripper_width],
            ]
        )

    def _is_grasp(self, actor, threshold: float = 1e-4, both_finger=False):
        all_contact = self.scene.get_contacts()
        robot_finger_links: List[LinkBase] = [
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
        # print(left_project_impulse, right_project_impulse)
        if both_finger:
            return (
                    left_project_impulse > threshold and right_project_impulse > threshold
            )
        else:
            return left_project_impulse > threshold or right_project_impulse > threshold
        
    def reset_arg(self, options_list=None):
        return self.reset()

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=True,
        action_relative="none", 
        allow_dir=["along"]  # zzt
    )
    obs = env.get_observation()

    imageio.imwrite(os.path.join("tmp", f"test1.jpg"), obs[f"third-rgb"])
    # imageio.imwrite(os.path.join("tmp", f"test2.jpg"), obs[f"forth-rgb"])
    print(env._get_gripper_width(), env._get_base_pose(), env._get_tcp_pose().p, quat2euler(env._get_tcp_pose().q))
    print(env.robot.get_qpos().copy()[env.arm_controller.arm_joint_indices])

    action = np.array([0.3, 0, 0.3, np.pi, 0, 0, 0.04])

    for i in range(30):
        env.step(action)

    obs = env.get_observation()
    imageio.imwrite(os.path.join("tmp", f"test2.jpg"), obs[f"third-rgb"])
    print(env._get_gripper_width(), env._get_base_pose(), env._get_tcp_pose().p, quat2euler(env._get_tcp_pose().q))
    print(env.robot.get_qpos().copy()[env.arm_controller.arm_joint_indices])


if __name__ == "__main__":
    test()
