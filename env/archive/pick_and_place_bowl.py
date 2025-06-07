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
    # load_lab_door,
    # generate_rand_door_config,
    load_lab_wall,
    # load_lab_scene_urdf,
    load_table_2,
    load_storage_box,
    load_blocks_on_table,
    build_actor_ycb,
    build_actor_egad,
    ASSET_DIR
)
from ..articulation.drawer_articulation import (
    load_table_4,
)
from ..robot import load_robot_panda
from ..controller.whole_body_controller import ArmSimpleController
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
        self.domain_randomize = domain_randomize
        self.canonical = canonical
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
        ycb_models_all = json.load(open(os.path.join(ASSET_DIR, "mani_skill2_ycb", "info_pick_v0.json"), "r"))
        # # self.model_db = ycb_models
        #
        # egad_models = json.load(open(os.path.join(ASSET_DIR, "mani_skill2_egad", "info_pick_train_v1.json"), "r"))
        # # self.model_db = egad_models
        # self.model_db = dict(
        #     ycb=ycb_models,
        #     egad=egad_models
        # )
        self.obj_infos = {"024_bowl": ycb_models_all["024_bowl"]}

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
        self.tray_cano_size = np.array([0.15, 0.12, 0.01])
        self.tray = load_storage_box(
            self.scene,
            box_size=self.tray_cano_size,
            thickness=0.003,
            color=np.array([0.6, 0.0, 0.0]),
            root_position=np.array([0.4, -0.2, self.table_top_z]),
            root_angle=0.
        )
        self.tray_info = dict(
            tray_size=self.tray_cano_size,
            tray_position=np.array([0.4, -0.2, self.table_top_z]),
            tray_yaw=0.
        )


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

    def reload_objs(self, obj_list=None):
        if hasattr(self, "objs"):
            for obj_id, obj_dict in self.objs.items():
                self.scene.remove_actor(obj_dict["actor"])
            self.objs.clear()
        else:
            self.objs = dict()

        for obj_id, obj_info in self.obj_infos.items():

            num_try = 0
            obj_invalid, init_p, init_angle, obj, init_scale = True, None, None, None, None
            while num_try < 10 and obj_invalid:
                rand_radius = self.np_random.uniform(0.03, 0.06)
                rand_ratio = self.np_random.uniform(0.9, 1.1)
                init_scale = np.array([
                    rand_radius / obj_info["bbox"]["max"][0],
                    rand_radius / obj_info["bbox"]["max"][1],
                    rand_radius * rand_ratio / (obj_info["bbox"]["max"][-1] * 2)
                ])

                bbox_min_z = obj_info["bbox"]["min"][-1] * init_scale[-1]

                rand_p = self.np_random.uniform(-1, 1, size=(2,))
                init_p = np.array([0.4 + 0.15 * rand_p[0], 0.3 * rand_p[1], self.table_top_z - bbox_min_z + 5e-3])

                init_angle = self.np_random.uniform(-np.pi, np.pi)

                dist = np.sqrt(((init_p[:2] - self.tray_info["tray_position"][:2]) ** 2).sum())
                if dist > self.tray_info["tray_size"][:2].max() + self.obj_infos[obj_id]["bbox"]["max"][0] * init_scale[0]:
                    obj_invalid = False

                num_try += 1
            # print(model_id, init_p, init_angle)

            model_type = "ycb"
            if not obj_invalid:
                if model_type == "ycb":
                    obj = build_actor_ycb(
                        obj_id, self.scene, root_position=init_p, root_angle=init_angle,
                        density=obj_info.get("density", 1000),
                        scale=init_scale
                    )
                    obj.set_damping(0.1, 0.1)
                    for body in obj.get_visual_bodies():
                        for rs in body.get_render_shapes():
                            # rs.material.set_base_color(np.concatenate([[0.7, 0.7, 0.7], [0.0]]))
                            rs.material.set_diffuse_texture_from_file("asset/pure_color_texture/white_08.jpg")
                else:
                    raise Exception("unknown data type!")

                self.objs.update({obj_id: dict(actor=obj, init_p=init_p, init_angle=init_angle, init_scale=init_scale)})

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

    def reset(self, seed: int = None, options: dict = None, obj_list: List = None):
        super().reset(seed, options)
        self.canonical_random, _ = seeding.np_random(seed)

        # Randomize properties in the beginning of episode
        if self.domain_randomize:
            # table
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

            # walls
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

            # tray
            tray_size_rand = self.np_random.uniform(-1, 1, (2,))
            tray_pose_rand = self.np_random.uniform(-1, 1, (3,))
            tray_rand_size = self.tray_cano_size + np.array([0.03 * tray_size_rand[0], 0.03 * tray_pose_rand[1], 0])
            tray_rand_position = np.array([0.4 + 0.1 * tray_pose_rand[0], 0.2 * tray_pose_rand[1], self.table_top_z])
            tray_rand_yaw = np.pi / 2 * tray_pose_rand[-1]
            if hasattr(self, "tray"):
                self.scene.remove_actor(self.tray)
            self.tray = load_storage_box(
                self.scene,
                box_size=tray_rand_size,
                thickness=0.003,
                color=np.array([0.6, 0.0, 0.0]),
                root_position=tray_rand_position,
                root_angle=tray_rand_yaw
            )
            self.tray_info = dict(
                tray_size=tray_rand_size,
                tray_position=tray_rand_position,
                tray_yaw=tray_rand_yaw
            )

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

                # tray
                for body in self.tray.get_visual_bodies():
                    for rs in body.get_render_shapes():
                        apply_random_texture(rs.material, self.canonical_random)

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

                # # storage box
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

        # reload objects
        self.reload_objs(obj_list=obj_list)

        if self.domain_randomize:
            for obj_id, obj_info in self.objs.items():
                obj = obj_info["actor"]
                for body in obj.get_visual_bodies():
                    for rs in body.get_render_shapes():
                        # rs.material.set_base_color(np.concatenate([[0.7, 0.7, 0.7], [0.0]]))
                        if self.canonical:
                            rs.material.set_diffuse_texture_from_file("asset/pure_color_texture/white_07.jpg")
                        else:
                            apply_random_texture(rs.material, self.canonical_random)

        # TODO: get obs
        self._update_observation()
        obs = OrderedDict()
        for key in self.obs_keys:
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
        # self.robot.set_drive_velocity_target(target_qvel)
        self.robot.set_qf(
            self.robot.compute_passive_force(
                external=False, coriolis_and_centrifugal=False
            )
        )
        for i in range(self.frame_skip):
            self.scene.step()

        # TODO: obs, reward, info
        # self._update_observation()  # step may skip update obs for faster rollout
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
        # print("In step, init_base_pose", self.init_base_pose)
        info.update(
            {
                "is_success": is_success,
                "init_base_pose": self.init_base_pose,
            }
        )
        return obs, reward, False, False, info

    def expert_action(self, obj_id, goal_obj_pose=None, noise_scale=0.0):
        # phases: before pregrasp, to grasp, close gripper, rotate, pull open
        actor = self.objs[obj_id]["actor"]
        init_p = self.objs[obj_id]["init_p"]
        init_angle = self.objs[obj_id]["init_angle"]
        init_scale = self.objs[obj_id]["init_scale"]
        obj_z_max = self.obj_infos[obj_id]["bbox"]["max"][-1] * init_scale[-1]
        done = False

        init_q = np.array(
            [np.cos(init_angle / 2), 0.0, 0.0, np.sin(init_angle / 2)]  # (w, x, y, z)
        )
        # print(obj_z_max, obj_id)

        # print(along)

        obj_pose = actor.get_pose()
        obj_init_pose = sapien.Pose(p=init_p, q=init_q)

        # print(obj_pose, obj_init_pose)

        obj_grasp_pose = obj_init_pose.transform(sapien.Pose(
            p=np.array([self.obj_infos[obj_id]["bbox"]["max"][0] * init_scale[0], 0, 0])
        ))
        obj_pre_grasp_pose = obj_grasp_pose.transform(sapien.Pose(
            p=np.array([0, 0, 0.1])
        ))

        tray_pose = self.tray.get_pose()
        obj_place_pose = sapien.Pose(
            p=tray_pose.p + np.array([0, 0, obj_z_max + 0.02]) + (obj_grasp_pose.p - obj_init_pose.p), q=obj_init_pose.q
        )
        obj_pre_place_pose = obj_place_pose.transform(sapien.Pose(
            p=np.array([0, 0, 0.1])
        ))

        obj_T_grasp = sapien.Pose.from_transformation_matrix(
            np.array(
                [
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ]
            )
        )

        # print("obj_pose", obj_pose, quat2euler(obj_pose.q))

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
            desired_grasp_pose = obj_pre_grasp_pose.transform(obj_T_grasp)
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
            # dp = sapien.Pose(p=desired_grasp_pose.p, q=euler2quat(np.array([np.pi, 0, 0])))
            # oTd = obj_pose.inv().transform(dp)
            # print(oTd, quat2euler(oTd.q), get_pose_from_rot_pos(quat2mat(oTd.q), oTd.p))
            # exit()

        elif self.expert_phase == 1:
            desired_grasp_pose = obj_grasp_pose.transform(obj_T_grasp)
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
            desired_grasp_pose = obj_grasp_pose.transform(obj_T_grasp)
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
            desired_grasp_pose = obj_pre_grasp_pose.transform(obj_T_grasp)
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

            desired_grasp_pose = obj_pre_place_pose.transform(obj_T_grasp)
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
            gripper_width = self._get_gripper_width()

            desired_grasp_pose = obj_place_pose.transform(obj_T_grasp)
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

        elif self.expert_phase == 6:
            desired_grasp_pose = obj_place_pose.transform(obj_T_grasp)
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

        elif self.expert_phase == 7:
            desired_grasp_pose = self.reset_tcp_pose
            desired_grasp_pose = grasp_pose_process(desired_grasp_pose)

            apply_noise_to_pose(desired_grasp_pose)
            desired_gripper_width = self.np_random.uniform(0, self.gripper_limit)

            # desired_gripper_width = (
            #         self.gripper_limit
            #         + self.np_random.uniform(-self.gripper_scale / 2, self.gripper_scale / 2) * noise_scale
            # )
            # desired_gripper_width = np.clip(desired_gripper_width, 0, self.gripper_limit)

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
            if (
                    np.linalg.norm(tcp_pose.p - desired_grasp_pose.p) < 0.01
                    and abs(qmult(tcp_pose.q, qconjugate(desired_grasp_pose.q))[0]) > 0.95
            ):
                self.expert_phase = 6

        elif self.expert_phase == 6:
            # pass
            if not self._is_grasp(actor):  # lost grasp, need to regrasp
                self.expert_phase = 7
        elif self.expert_phase == 7:
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
        # if self.agv_link_idx is None:
        #     for link_idx, link in enumerate(self.robot.get_links()):
        #         if link.get_name() == "mount_link_fixed":
        #             self.agv_link_idx = link_idx
        # link = self.robot.get_links()[self.agv_link_idx]
        # return link.get_pose()

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
        domain_randomize=True,
        canonical=False
    )

    cano_pick_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        # obs_keys=("wrist-rgb", "tcp_pose", "gripper_width"),
        obs_keys=(),
        domain_randomize=True,
        canonical=True
    )

    # envs = [rand_pick_env, cano_pick_env]
    envs = {
        "rand": rand_pick_env,
        "cano": cano_pick_env,
    }
    cameras = ["third"]

    save_dir = "/root/data/sim2sim_pd_1"
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

                try:

                    goal_p_rand = random_state.uniform(-0.1, 0.1, size=(2,))
                    goal_q_rand = random_state.uniform(-0.5, 0.5)

                    prev_privileged_obs = None
                    for step in range(steps_per_obj):

                        action, done, _ = env.expert_action(
                            noise_scale=0.2, obj_id=model_id,
                            goal_obj_pose=sapien.Pose(
                                p=np.concatenate([np.array([0.4, -0.2]) + goal_p_rand, [0.76]]),
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

                except Exception as e:
                    print(seed, env_key, model_id, e)
                    env.expert_phase = 0

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
        obs_keys=("tcp_pose", "gripper_width"),
        # obs_keys=(),
        domain_randomize=True,
        canonical=True,
        # canonical=False,
    )

    env = cano_pick_env
    cameras = ["third"]

    save_dir = "/root/data/cano_bowl_1007"
    # save_dir = "try"
    num_seeds = 5000
    # num_seeds = 3
    num_vid = 10
    os.makedirs(save_dir, exist_ok=True)

    num_suc = 0
    success_list = []

    from tqdm import tqdm

    for seed in tqdm(range(num_seeds)):
        save_path = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(save_path, exist_ok=True)

        env.reset(seed=seed)

        # random_state = np.random.RandomState(seed=seed)

        if seed < num_vid:
            video_writer = {cam: imageio.get_writer(
                f"tmp/seed_{seed}_cam_{cam}.mp4",
                # fps=40,
                fps=20,
                format="FFMPEG",
                codec="h264",
            ) for cam in cameras}

        success = False
        frame_id = 0

        try:
            model_id = list(env.objs.keys())[0]
            prev_privileged_obs = None
            while True:
                action, done, desired_dict = env.expert_action(
                    noise_scale=0.2, obj_id=model_id,
                )
                _, _, _, _, info = env.step(action)

                # rgb_images = env.render_all()
                # rgb_images = env.capture_images_new(cameras=cameras)
                if frame_id < 400:
                    if done:
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

                    for cam in cameras:
                        image = obs.pop(f"{cam}-rgb")
                        imageio.imwrite(os.path.join(save_path, f"step_{frame_id}_cam_{cam}.jpg"), image)
                        if seed < num_vid:
                            video_writer[cam].append_data(image)

                    pickle.dump(obs, open(os.path.join(save_path, f"step_{frame_id}.pkl"), "wb"))
                    frame_id += 1

                else:
                    break

        except Exception as e:
            print("error: ", seed, e)

        if success:
            success_list.append((seed, "s", frame_id))
            # print(seed, "s", frame_id)

            num_suc += 1
        else:
            success_list.append((seed, "f", frame_id))
            # print(seed, "f", frame_id)

        if seed < num_vid:
            for writer in video_writer.values():
                writer.close()

    pickle.dump(success_list, open(os.path.join(save_dir, f"info.pkl"), "wb"))

    print(num_suc)


def collect_sim2sim_data():
    from env.sapien_utils.wrapper import StateObservationWrapper, TimeLimit

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cano_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=True,
        # action_relative="none"
    )

    rand_env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=False,
        # action_relative="none"
    )

    cameras = ["third"]

    save_dir = "/root/data/sim2sim_bowl_1004"
    # save_dir = "try"
    num_seeds = 10000
    # num_seeds = 10
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
            model_id = list(cano_env.objs.keys())[0]
            prev_privileged_obs = None

            while True:
                action, done, desired_dict = cano_env.expert_action(
                    noise_scale=0.2, obj_id=model_id
                )

                _, _, _, _, info = cano_env.step(action)
                _, _, _, _, info = rand_env.step(action)

                if frame_id < 400:
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


# def eval_imitation():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     env = PickAndPlaceEnv(
#         use_gui=False,
#         device=device,
#         # obs_keys=("wrist-rgb", "tcp_pose", "gripper_width"),
#         obs_keys=(),
#         domain_randomize=True,
#         canonical=True,
#         action_relative="none"
#     )
#
#     cameras = ["third", "wrist"]
#     usage = ["obs"]
#
#     stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#
#     save_dir = os.path.join("tmp", stamp)
#     # save_dir = "try"
#     # num_seeds = 10000
#     num_eval = 10
#     os.makedirs(save_dir, exist_ok=True)
#
#     # cnt_list = []
#
#     from tqdm import tqdm
#
#     for i_eval in range(num_eval):
#         seed = i_eval + 1000
#         save_path = os.path.join(save_dir, f"seed_{seed}")
#         os.makedirs(save_path, exist_ok=True)
#
#         env.reset(seed=seed)
#
#         random_state = np.random.RandomState(seed=seed)
#
#         model_id_list = list(env.objs.keys())
#         # print(model_id_list)
#         random_state.shuffle(model_id_list)
#
#         for ep_id, model_id in enumerate(model_id_list):
#
#             ep_path = os.path.join(save_path, f"ep_{ep_id}")
#             os.makedirs(ep_path, exist_ok=True)
#
#             video_writer = {cam: imageio.get_writer(
#                 os.path.join(ep_path, f"seed_{seed}_ep_{ep_id}_cam_{cam}.mp4"),
#                 fps=20,
#                 format="FFMPEG",
#                 codec="h264",
#             ) for cam in cameras}
#
#             for step in tqdm(range(500)):
#                 obs = env.get_observation()
#                 for cam in cameras:
#                     imageio.imwrite(os.path.join(ep_path, f"obs-{cam}.jpg"), obs[f"{cam}-rgb"])
#                     video_writer[cam].append_data(obs[f"{cam}-rgb"])
#
#                 files = {
#                     f"{usg}-{cam}": open(os.path.join(ep_path, f"{usg}-{cam}.jpg"), "rb")
#                     for cam in cameras for usg in usage
#                 }
#
#                 pose = obs["tcp_pose"]
#                 pose_p, pose_q = pose[:3], pose[3:]
#                 pose_mat = quat2mat(pose_q)
#
#                 pose_at_obs = get_pose_from_rot_pos(
#                     pose_mat, pose_p
#                 )
#
#                 pose_mat_6 = pose_mat[:, :2].reshape(-1)
#                 proprio_state = np.concatenate(
#                     [
#                         pose_p,
#                         pose_mat_6,
#                         np.array([obs["gripper_width"]]),
#                     ]
#                 )
#                 data = {"proprio_state": json.dumps(proprio_state.tolist())}
#
#                 response = requests.post("http://localhost:9977/diffusion", files=files, data=data)
#                 response = response.json()
#                 # print(response)
#
#                 actions = np.array(response["action"])
#                 converted_actions = []
#
#                 for i in range(10):
#                     action = actions[i]
#                     mat_6 = action[3:9].reshape(3, 2)
#                     mat_6[:, 0] = mat_6[:, 0] / np.linalg.norm(mat_6[:, 0])
#                     mat_6[:, 1] = mat_6[:, 1] / np.linalg.norm(mat_6[:, 1])
#                     z_vec = np.cross(mat_6[:, 0], mat_6[:, 1])
#                     mat = np.c_[mat_6, z_vec]
#                     # assert mat.shape == (3, 3)
#
#                     pos = action[:3]
#                     gripper_width = action[-1]
#
#                     init_to_desired_pose = pose_at_obs @ get_pose_from_rot_pos(
#                         mat, pos
#                     )
#                     pose_action = np.concatenate(
#                         [
#                             [0, 0],
#                             init_to_desired_pose[:3, 3],
#                             mat2euler(init_to_desired_pose[:3, :3]),
#                             [gripper_width]
#                         ]
#                     )
#                     converted_actions.append(pose_action)
#
#                     _, _, _, _, info = env.step(pose_action)
#
#             for writer in video_writer.values():
#                 writer.close()
#
#             # exit()


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = PickAndPlaceEnv(
        use_gui=False,
        device=device,
        obs_keys=(),
        domain_randomize=True,
        canonical=True,
        action_relative="none"
    )
    obs = env.get_observation()

    imageio.imwrite(os.path.join("tmp", f"test1.jpg"), obs[f"third-rgb"])
    # imageio.imwrite(os.path.join("tmp", f"test2.jpg"), obs[f"forth-rgb"])
    print(env._get_gripper_width(), env._get_base_pose(), env._get_tcp_pose().p, quat2euler(env._get_tcp_pose().q))
    print(env.robot.get_qpos().copy()[env.arm_controller.arm_joint_indices])

    action = np.array([0.3, 0, 0.3, np.pi, 0, 0, 0.04])

    for _ in range(30):
        env.step(action)

    obs = env.get_observation()
    imageio.imwrite(os.path.join("tmp", f"test2.jpg"), obs[f"third-rgb"])

    print(env._get_gripper_width(), env._get_base_pose(), env._get_tcp_pose().p, quat2euler(env._get_tcp_pose().q))
    print(env.robot.get_qpos().copy()[env.arm_controller.arm_joint_indices])

    # white = np.ones_like(obs[f"third-rgb"])
    #
    # for i in range(11):
    #     w = np.around(white * (i / 10) * 255).astype(np.uint8)
    #     imageio.imwrite("asset/pure_color_texture/white_{:02d}.jpg".format(i), w)


if __name__ == "__main__":
    # collect_sim2sim_data()
    collect_imitation_data()
    # eval_imitation()
    # test()
