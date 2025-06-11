import cv2
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
import os
import sapien.core as sapien
import torch
import imageio
from collections import OrderedDict
from env.sapien_utils.base import BaseEnv, recover_action, get_pairwise_contact_impulse, get_pairwise_contacts

# from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat
from typing import List
from env.sapien_utils.math import wrap_to_pi, euler2quat, quat2euler, mat2euler, get_pose_from_rot_pos

from env.utils import apply_random_texture, check_intersect_2d, grasp_pose_process, check_intersect_2d_
from env.articulation.pick_and_place_articulation import (
    # load_lab_door,
    # generate_rand_door_config,
    load_lab_wall,
    # load_lab_scene_urdf,
    load_table_2,
    load_storage_box,
    load_blocks_on_table,
    ASSET_DIR
)
from env.articulation.drawer_articulation import (
    load_drawer_urdf,
    load_drawers,
    load_table_4,
)
from env.robot import load_robot_panda
from env.controller.whole_body_controller import ArmSimpleController, PandaSimpleController
import json
import pickle
import requests
from datetime import datetime


class DrawerPushAndPullEnv(BaseEnv):
    def __init__(
            self,
            use_gui: bool,
            device: str,
            mipmap_levels=1,
            obs_keys=tuple(),
            action_relative="tool",
            domain_randomize=True,
            canonical=True, 
            use_image_obs=False
    ):
        self.target_open_amount = 0.05
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
        self.use_image_obs=use_image_obs
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
            # fov=np.deg2rad(60),
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
        self.arm_controller = PandaSimpleController(self.robot)
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

        self.robot, self.finger_link_idxs = load_robot_panda(self.scene)

        self.table_top_z = 0.76

        self.drawer_scale = 0.25
        self.drawer_base_z = 0.16 * self.drawer_scale
        # self.storage_box = load_storage_box(self.scene, root_position=np.array([0.4, -0.2, self.table_top_z]))
        drawer_poses = [
            sapien.Pose(
                p=np.array([0.4, -0.3, self.table_top_z + self.drawer_base_z]),
                q=euler2quat(np.array([0, 0, np.pi / 2]))
            ),
            sapien.Pose(
                p=np.array([0.4, -0.3, self.table_top_z + self.drawer_base_z * 3]),
                q=euler2quat(np.array([0, 0, np.pi / 2]))
            )
        ]
        self.drawers = load_drawers(self.scene, self.drawer_scale, drawer_poses)


    
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
        self.has_open = False
        self.has_close_after_open = False
        self.is_success = False
        self.canonical_random, _ = seeding.np_random(seed)

        # Randomize properties in the beginning of episode
        if self.domain_randomize:
            table_rand = self.np_random.uniform(-1, 1, (2,))
            table_rand_size = np.array([0.1 * table_rand[0], 0.1 * table_rand[1], 0.]) + np.array([0.8, 1.0, 0.03])
            leg_pos_x = table_rand_size[0] / 2 - 0.1
            leg_pos_y = table_rand_size[1] / 2 - 0.1
            table_position = np.array([0.23, 0, 0])
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

            # drawer
            drawer_rand = self.np_random.uniform(-1, 1, (3,))
            yaw_range = (np.pi / 3, np.pi * 3 / 5)
            yaw_rand = (yaw_range[1] + yaw_range[0]) / 2 + drawer_rand[-1] * (yaw_range[1] - yaw_range[0]) / 2
            if hasattr(self, "drawers"):
                for drawer in self.drawers:
                    self.scene.remove_articulation(drawer)
            drawer_poses = [
                sapien.Pose(
                    p=np.array([0.4, -0.3, self.table_top_z + self.drawer_base_z]) +
                      np.array([drawer_rand[0], drawer_rand[1], 0]) * np.array([0.05, 0.05, 0]),
                    # p=np.array([0.6, -0.1, 1.2]),
                    q=euler2quat(np.array([0, 0, yaw_rand]))
                ),
                sapien.Pose(
                    p=np.array([0.4, -0.3, self.table_top_z + self.drawer_base_z * 3]) +
                      np.array([drawer_rand[0], drawer_rand[1], 0]) * np.array([0.05, 0.05, 0]),
                    # p=np.array([0.6, -0.1, 1.2]),
                    q=euler2quat(np.array([0, 0, yaw_rand]))
                )
            ]
            self.drawers = load_drawers(self.scene, self.drawer_scale, drawer_poses)

            # self.drawer = load_drawer_urdf(self.scene, scale=self.drawer_scale)
            # self.drawer.set_root_pose(
            #     sapien.Pose(
            #         p=np.array([0.4, -0.3, self.table_top_z + self.drawer_base_z]) +
            #         np.array([drawer_rand[0], drawer_rand[1], 0]) * np.array([0.05, 0.05, 0]),
            #         # p=np.array([0.6, -0.1, 1.2]),
            #         q=euler2quat(np.array([0, 0, yaw_rand]))
            #     )
            # )

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

                # drawer
                for drawer in self.drawers:
                    for link_idx, link in enumerate(drawer.get_links()):
                        vb_list = link.get_visual_bodies()
                        for vb in vb_list:
                            for rs in vb.get_render_shapes():
                                apply_random_texture(rs.material, self.canonical_random)
                #         if body.name in ["upper_surface", "bottom_surface"]:
                #             apply_random_texture(rs.material, self.canonical_random)
                #         else:
                #             storage_box_materials.append(rs.material)
                # apply_random_texture(storage_box_materials, self.canonical_random)

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
                # for body in self.storage_box.get_visual_bodies():
                #     for rs in body.get_render_shapes():
                #         if body.name in ["upper_surface", "bottom_surface"]:
                #             rs.material.set_base_color(
                #                 np.array(
                #                     [0.4, 0.2, 0.0, 1.0]
                #                 )
                #             )
                #         else:
                #             rs.material.set_base_color(
                #                 np.array(
                #                     [0.6, 0.3, 0.0, 1.0]
                #                 )
                #             )

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
                self.standard_head_cam_fovx + self.np_random.uniform(*fov_rand_range),
                compute_y=True,
            )
            # self.cameras["wrist"].set_fovx(
            #     self.standard_wrist_cam_fovx + self.np_random.uniform(-0.1, 0.1),
            #     compute_y=True,
            # )
            # 取消相机随机变换，直接设置为标准相机位姿与视角
            self.cameras["third"].set_pose(self.standard_head_cam_pose)
            self.cameras["third"].set_fovx(self.standard_head_cam_fovx, compute_y=True)

        init_p = self.np_random.uniform(
            low=np.array([-0.01, -0.02, 0.78]), high=np.array([0.01, 0.02, 0.78])
        )

        init_angle = self.np_random.uniform(
            low=-0.01, high=0.01
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


        # TODO: get obs
        self._update_observation()
        obs = OrderedDict()
        for key in self.observation_dict:
            obs[key] = self.observation_dict[key]

        return obs, {}

    def step(self, action: np.ndarray):
        action = action.copy()
        info = {}

        if len(action) == 7:
            # TODO: IK for arm, velocity control for mobile base
            # action is desired base w, desired base v, delta eef pose, binary gripper action in the specified frame
            cur_ee_pose = self._get_tcp_pose()
            cur_relative_ee_pose = self.init_base_pose.inv().transform(cur_ee_pose)
            # action relative to tool frame
            if self.action_relative == "tool":
                desired_relative_ee_pose = cur_relative_ee_pose.transform(
                    sapien.Pose(
                        p=action[:3],
                        q=euler2quat(action[3:6]),
                    )
                )
            # action relative to fixed frame
            elif self.action_relative == "base":
                desired_relative_ee_pose = sapien.Pose(
                    cur_relative_ee_pose.p + self.p_scale * action[:3],
                    euler2quat(
                        quat2euler(cur_relative_ee_pose.q)
                        + self.rot_scale * action[3:6]
                    ),
                )
                # pose in initial agv frame
            elif self.action_relative == "none":
                desired_relative_ee_pose = sapien.Pose(
                    p=action[:3], q=euler2quat(action[3:6])
                )
                # dirty
                # action[8] = -action[8]
            desired_ee_pose = self.init_base_pose.transform(desired_relative_ee_pose)

            target_qpos, control_success = self.arm_controller.compute_q_target(
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
            control_success = True

        info["desired_joints"] = target_qpos[
            self.arm_controller.arm_joint_indices
        ]
        info["desired_gripper_width"] = (
            target_qpos[self.arm_controller.finger_joint_indices[0]]
        )

        self.robot.set_drive_target(target_qpos)
        # self.robot.set_drive_velocity_target(target_qvel)
        self.robot.set_qf(
            self.robot.compute_passive_force(
                external=False, coriolis_and_centrifugal=False
            )
        )
        for i in range(self.frame_skip):
            self.scene.step()

        # TODO: obs, reward, info
        # print(self.objs)
        self._update_observation()
        obs = OrderedDict()
        for key in self.observation_dict:
            obs[key] = self.observation_dict[key]

        reward = self.compute_dense_reward()
        info.update(
            {
                "is_success": self.is_success,
                "init_base_pose": self.init_base_pose,
            }
        )
        return obs, reward, False, False, info
        # return obs, reward, self.is_success, False, info

    def compute_dense_reward(self, drawer_id=1):
        reward = 0.0
        
        # reaching reward
        for link_idx, link in enumerate(self.drawers[drawer_id].get_links()):
            if link.get_name() == "handle":
                handle_pose = link.get_pose()
        
        tcp_pose = self._get_tcp_pose()
        
        tcp_to_handle_dist = np.linalg.norm(tcp_pose.p - handle_pose.p)
        reaching_reward = 1.0 - np.tanh(5.0 * tcp_to_handle_dist)
        # reward += reaching_reward # Delete reaching reward
        
        # open reward
        if not self.has_open:
            open_amount = max(0, self.drawers[drawer_id].get_qpos()[0])
            # open_reward = 2 * (1 - (self.target_open_amount - open_amount))
            open_reward = 0 # Only stage reward
            
            if open_amount > self.target_open_amount:
                open_reward = 2
                self.has_open = True
            reward += open_reward
            
        # close reward
        if self.has_open and not self.has_close_after_open:
            reward += 2 # for open reward
            open_amount = max(0, self.drawers[drawer_id].get_qpos()[0])
            # close_reward = 2 * (1 - open_amount) 
            close_reward = 0 # Only stage reward
            if open_amount < 1e-3:
                close_reward = 2
                self.has_close_after_open = True
            reward += close_reward
        
        if self.has_open and self.has_close_after_open:
            self.is_success = True
            reward = 10
        
        reward = reward / 10.0

        return reward
        
        
        
    def expert_action(self, obj_id=None, goal_obj_pose=None, noise_scale=0.0, drawer_id=1):

        drawer_pose = self.drawers[drawer_id].get_pose()

        handle_pose = drawer_pose.transform(sapien.Pose(
            p=np.array([0.345, 0.0, 0.0]) * self.drawer_scale + np.array([0.01, 0, 0])
        ))
        push_handle_pose = drawer_pose.transform(sapien.Pose(
            p=np.array([0.345, 0.0, 0.0]) * self.drawer_scale
        ))

        pre_grasp_pose = handle_pose.transform(sapien.Pose(
            p=np.array([0.1, 0.0, 0.0])
        ))
        post_grasp_pose = handle_pose.transform(sapien.Pose(
            p=np.array([0.1, 0.0, 0.0])
        ))
        release_pose = handle_pose.transform(sapien.Pose(
            p=np.array([0.2, 0.0, 0.0])
        ))
        # print(drawer_pose, handle_pose)

        # print(drawer_pose, tcp_pose)
        handle_T_grasp = sapien.Pose.from_transformation_matrix(
            np.array(
                [
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [-1, 0, 0, 0],
                    [0, 0, 0, 1],
                ]
            )
        )

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

            desired_grasp_pose = pre_grasp_pose.transform(handle_T_grasp)
            # print("desired_grasp_pose", desired_grasp_pose, quat2euler(desired_grasp_pose.q))
            apply_noise_to_pose(desired_grasp_pose)

            # randomize gripper width in phase 0
            desired_gripper_width = self.np_random.uniform(0, self.gripper_limit)
            # desired_gripper_width = self.gripper_limit + self.np_random.uniform(-0.02, 0.02) * noise_scale

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )

        elif self.expert_phase == 1:
            desired_grasp_pose = handle_pose.transform(handle_T_grasp)
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
            desired_grasp_pose = handle_pose.transform(handle_T_grasp)

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

            desired_grasp_pose = post_grasp_pose.transform(handle_T_grasp)

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

            desired_grasp_pose = release_pose.transform(handle_T_grasp)

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

        elif self.expert_phase == 5:
            desired_grasp_pose = push_handle_pose.transform(handle_T_grasp)

            apply_noise_to_pose(desired_grasp_pose)

            desired_gripper_width = self.np_random.uniform(0, self.gripper_limit)

            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                desired_gripper_width,
            )

        elif self.expert_phase == 6:
            # desired_grasp_pose = self.reset_tcp_pose
            desired_grasp_pose = release_pose.transform(handle_T_grasp)

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
        done = False
        # print(tcp_pose, "tcp")

        if (
                np.linalg.norm(tcp_pose.p - desired_grasp_pose.p) < 0.01
                and abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]) > 0.95
        ):
            if self.expert_phase == 6:
                self.expert_phase = 0
                done = True
            elif self.expert_phase == 2:
                # print(gripper_width, self.gripper_scale, desired_gripper_width)
                if gripper_width < self.gripper_scale:
                    self.expert_phase += 1
            else:
                self.expert_phase += 1

        return action, done, {"desired_grasp_pose": desired_grasp_pose,
                              "desired_gripper_width": desired_gripper_width}

    # compute all the observations
    def _update_observation(self):
        self.observation_dict.clear()
        if self.use_image_obs:
            image_obs = self.capture_images_new()
        world_tcp_pose = self._get_tcp_pose()
        tcp_pose = self.init_base_pose.inv().transform(self._get_tcp_pose())
        gripper_width = self._get_gripper_width()
        # arm_joints = self.robot.get_qpos()[self.arm_controller.arm_joint_indices]

        if self.use_image_obs:
            self.observation_dict.update(image_obs)
        self.observation_dict["tcp_pose"] = np.concatenate([tcp_pose.p, tcp_pose.q])
        self.observation_dict["gripper_width"] = gripper_width
        # self.observation_dict["robot_joints"] = arm_joints
        self.observation_dict["privileged_obs"] = np.concatenate(
            [
                world_tcp_pose.p,
                world_tcp_pose.q,
                [gripper_width]
            ]
        )
        # self.observation_dict["obj_states"] = obj_states

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

    def _get_tcp_pose(self) -> sapien.Pose:
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

    def _get_base_pose(self) -> sapien.Pose:
        return self.robot.get_pose()

    def _desired_tcp_to_action(
            self,
            tcp_pose: sapien.Pose,
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
        # print(left_project_impulse, right_project_impulse)
        if both_finger:
            return (
                    left_project_impulse > threshold and right_project_impulse > threshold
            )
        else:
            return left_project_impulse > threshold or right_project_impulse > threshold

    def _is_success(self, actor):
        # obj_pose = actor.get_pose()
        # all_contact = self.scene.get_contacts()
        # contacts = get_pairwise_contacts(all_contact, actor, self.storage_box)
        #
        # if 0.25 < obj_pose.p[0] < 0.55 and -0.35 < obj_pose.p[1] < -0.05 and len(contacts):
        #     success = True
        # else:
        #     success = False
        #
        # return success
        return False



def test():

    from homebot_sapien.utils.wrapper import StateObservationWrapper, TimeLimit

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = DrawerPushAndPullEnv(
        use_gui=False,
        device=device,
        obs_keys=("tcp_pose", "gripper_width"),
        domain_randomize=True,
        canonical=True,
        # action_relative="none"
    )
    env_wrapper = StateObservationWrapper(TimeLimit(env))

    env_wrapper.reset()

    obs = env_wrapper.env.get_observation()
    imageio.imwrite(os.path.join("tmp", f"test1.jpg"), obs[f"third-rgb"])

    for _ in range(10):
        action = np.random.uniform(-1, 1, size=(10,))
        o, r, d, t, i = env_wrapper.step(action)
        print(o, r, d)

    obs = env_wrapper.env.get_observation()
    imageio.imwrite(os.path.join("tmp", f"test2.jpg"), obs[f"third-rgb"])

def test_expert_grasp():
    from homebot_sapien.utils.wrapper import StateObservationWrapper, TimeLimit
    # stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = DrawerPushAndPullEnv(
        use_gui=False,
        device=device,
        obs_keys=("tcp_pose", "gripper_width"),
        domain_randomize=True,
        canonical=True,
        # action_relative="none"
    )
    env_wrapper = StateObservationWrapper(TimeLimit(env))
    cameras = ["third"]

    num_seeds = 10  # cano test
    num_vid = 10

    num_suc = 0
    success_list = []

    from tqdm import tqdm

    for seed in tqdm(range(num_seeds)):

        env_wrapper.env.reset(seed=seed)
        video_dir = f"tmp/drawer/"  #{stamp}
        os.makedirs(video_dir, exist_ok=True)

        if seed < num_vid:
            video_writer = {cam: imageio.get_writer(
                f"{video_dir}/seed_{seed}_drawer.mp4",
                fps=20,
                format="FFMPEG",
                codec="h264",
            ) for cam in cameras}

        success = False
        frame_id = 0

        try:
            prev_privileged_obs = None
            while True:
                action, done, desired_dict = env_wrapper.env.expert_action(
                    noise_scale=0.2,
                )

                o, _, _, _, info = env_wrapper.env.step(action)
                o = env_wrapper.process_obs(o)

                if frame_id < 500:
                    if done:
                        print("done", "frame_id", frame_id)
                        success = True
                        break
                    obs = env_wrapper.env.get_observation()
                    obs.update({"action": action})
                    obs.update(desired_dict)
                    obs.update({"wrapper_obs": o})

                    if prev_privileged_obs is not None and np.all(
                            np.abs(obs["privileged_obs"] - prev_privileged_obs) < 1e-4):
                        env_wrapper.env.expert_phase = 0
                        break
                    prev_privileged_obs = obs["privileged_obs"]

                    for cam in cameras:
                        image = obs.pop(f"{cam}-rgb")
                        if seed < num_vid:
                            video_writer[cam].append_data(image)

                    frame_id += 1
                else:
                    break

        except Exception as e:
            print("error: ", seed, e)

        if success:
            success_list.append((seed, "s", frame_id))
            num_suc += 1
        else:
            success_list.append((seed, "f", frame_id))

        if seed < num_vid:
            for writer in video_writer.values():
                writer.close()

    # with open("success_list.txt", "w") as f:
    #     for entry in success_list:
    #         f.write(f"{entry[0]} {entry[1]} {entry[2]}\n")
    print(success_list)
    print(num_suc)

def collect_sim2sim_data():
    from homebot_sapien.utils.wrapper import StateObservationWrapper, TimeLimit

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cano_env = DrawerPushAndPullEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=True,
        # action_relative="none"
    )

    rand_env = DrawerPushAndPullEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=False,
        # action_relative="none"
    )

    cameras = ["third"]

    save_dir = "./tmp/data/sim2sim_drawer"
    # save_dir = "try"
    num_seeds = 10000
    num_vid = 10
    os.makedirs(save_dir, exist_ok=True)

    num_suc = 0
    success_list = []

    from tqdm import tqdm

    for seed in tqdm(range(num_seeds)):
        save_path = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(save_path, exist_ok=True)

        cano_env.reset(seed=seed)
        rand_env.reset(seed=seed)

        if seed < num_vid:
            cano_video_writer = {cam: imageio.get_writer(
                f"tmp/seed_{seed}_cam_{cam}_cano.mp4",
                # fps=40,
                fps=20,
                format="FFMPEG",
                codec="h264",
            ) for cam in cameras}

            rand_video_writer = {cam: imageio.get_writer(
                f"tmp/seed_{seed}_cam_{cam}_rand.mp4",
                # fps=40,
                fps=20,
                format="FFMPEG",
                codec="h264",
            ) for cam in cameras}

        success = False
        frame_id = 0

        try:
            prev_privileged_obs = None
            while True:
                action, done, desired_dict = cano_env.expert_action(
                    noise_scale=0.2,
                )

                _, _, _, _, info = cano_env.step(action)
                _, _, _, _, info = rand_env.step(action)

                if frame_id < 500:
                    if done:
                        success = True
                        break

                    if frame_id % 10 == 0:
                        cano_obs = cano_env.get_observation()
                        rand_obs = rand_env.get_observation()

                        assert np.abs(cano_obs["privileged_obs"] - rand_obs["privileged_obs"]).all() < 1e-4, \
                            "cano and rand envs not match!"

                        if prev_privileged_obs is not None and np.all(
                                np.abs(cano_obs["privileged_obs"] - prev_privileged_obs) < 1e-4):
                            cano_env.expert_phase = 0
                            break
                        prev_privileged_obs = cano_obs["privileged_obs"]

                        for cam in cameras:
                            cano_image = cano_obs.pop(f"{cam}-rgb")
                            rand_image = rand_obs.pop(f"{cam}-rgb")
                            imageio.imwrite(os.path.join(save_path, f"step_{frame_id}_cam_{cam}_cano.jpg"), cano_image)
                            imageio.imwrite(os.path.join(save_path, f"step_{frame_id}_cam_{cam}_rand.jpg"), rand_image)
                            if seed < num_vid:
                                cano_video_writer[cam].append_data(cano_image)
                                rand_video_writer[cam].append_data(rand_image)

                    # pickle.dump(obs, open(os.path.join(save_path, f"step_{frame_id}.pkl"), "wb"))
                    frame_id += 1

                else:
                    break

        except Exception as e:
            print("error: ", seed, e)

        if success:
            success_list.append((seed, "s", frame_id))
            num_suc += 1
            # print(seed, "s", frame_id)
        else:
            success_list.append((seed, "f", frame_id))
            # print(seed, "f", frame_id)

        if seed < num_vid:
            for writer in cano_video_writer.values():
                writer.close()
            for writer in rand_video_writer.values():
                writer.close()

    pickle.dump(success_list, open(os.path.join(save_dir, f"info.pkl"), "wb"))

    print(num_suc)


if __name__ == "__main__":
    # test()
    test_expert_grasp()
    # collect_sim2sim_data()

