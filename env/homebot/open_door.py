import cv2
import gymnasium as gym
import numpy as np
import os
import sapien.core as sapien
import torch
import imageio
from collections import OrderedDict
from env.sapien_utils.base import BaseEnv, recover_action , get_pairwise_contact_impulse

# from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qmult, qconjugate
from typing import List
from env.sapien_utils.math import wrap_to_pi, euler2quat, quat2euler
from env.utils import apply_random_texture
from env.articulation.door_articulation import (
    load_lab_door,
    generate_rand_door_config,
    load_lab_wall,
    load_lab_scene_urdf,
)
from env.robot import load_robot_full
from env.controller.whole_body_controller import BaseArmSimpleController


class OpenDoorEnv(BaseEnv):
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
        use_image_obs=False, 
        **unused
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
        self.use_image_obs = use_image_obs
        super().__init__(use_gui, device, mipmap_levels)
        if not self.door_from_urdf:
            if use_real:
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
                cam_position = np.array([-0.7, -0.3, 1.4])
                look_at_dir = np.array([1.0, 0.7, -0.3])
                right_dir = np.array([0.7, -1.0, 0.0])
                self.create_camera(
                    position=cam_position,
                    # position=np.array([-1.7, -0.3, 1.0]),
                    look_at_dir=look_at_dir,
                    right_dir=right_dir,
                    name="third",
                    resolution=(320, 240),
                    fov=np.deg2rad(60),
                )

        else:
            self.create_camera(
                position=np.array([-1.2, -0.3, 1.4]),
                look_at_dir=np.array([1.0, 0.7, -0.3]),
                right_dir=np.array([0.7, -1.0, 0.0]),
                name="third",
                resolution=(320, 240),
                fov=np.deg2rad(50),  
            )
        self.standard_head_cam_pose = self.cameras["third"].get_pose()
        self.standard_head_cam_fovx = self.cameras["third"].fovx
        camera_mount_actor = self.robot.get_links()[-1]
        self.create_camera(
            None, None, None, "wrist", (320, 240), np.deg2rad(50), camera_mount_actor
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
        self.reset()
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
        # self.door_articulation = self.load_door()
        if not self.door_from_urdf:
            self.door_articulation = load_lab_door(
                self.scene,
                generate_rand_door_config(self.np_random, use_real=self.use_real),
                self.np_random,
                need_door_shut=self.need_door_shut,
            )
        else:
            self.door_articulation = load_lab_scene_urdf(self.scene)
        # self.robot = self.load_robot()
        self.robot, self.finger_link_idxs = load_robot_full(self.scene)
        if "free" in self.robot.get_name():
            qpos = self.robot.get_qpos()
            # NOTE: the rotation dofs are not the euler angles
            qpos[:6] = np.array([-0.5, 0.0, 0.5, np.pi, np.pi / 3, 0.0])
            self.robot.set_qpos(qpos)

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
        if not self.door_from_urdf:
            # Add room walls
            self.room_wall1 = load_lab_wall(self.scene)

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed, options)
        # Randomize properties in the beginning of episode
        if self.domain_randomize:
            self.scene.set_ambient_light(np.tile(self.np_random.uniform(0, 1, (1,)), 3))
            self.scene.remove_light(self.light0)
            self.light0 = self.scene.add_directional_light(
                self.np_random.uniform(-1, 1, (3,)),
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
            self.room_ground.get_visual_bodies()[0].get_render_shapes()[
                0
            ].material.set_base_color(
                np.concatenate(
                    [np.tile(self.np_random.uniform(0, 1, size=(1,)), 3), [1.0]]
                )
            )
            if not self.door_from_urdf:
                self.scene.remove_actor(self.room_wall1)
                self.room_wall1 = load_lab_wall(self.scene)
                for rs in self.room_wall1.get_visual_bodies()[0].get_render_shapes():
                    apply_random_texture(rs.material, self.np_random)
                self.scene.remove_articulation(self.door_articulation)
                door_config = generate_rand_door_config(
                    self.np_random, use_real=self.use_real
                )
                self.door_articulation = load_lab_door(
                    self.scene,
                    door_config,
                    self.np_random,
                    need_door_shut=self.need_door_shut,
                )
            else:
                self.scene.remove_articulation(self.door_articulation)
                self.door_articulation = load_lab_scene_urdf(self.scene)
                self.door_articulation.set_root_pose(
                    sapien.Pose(np.array([0.0, 0.0, 0.8467]))
                )
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
            # self.cameras["wrist"].set_fovx(
            #     self.standard_wrist_cam_fovx + self.np_random.uniform(-0.05, 0.05),
            #     compute_y=True,
            # )

            # 取消相机随机变换，直接设置为标准相机位姿与视角
            self.cameras["third"].set_pose(self.standard_head_cam_pose)
            self.cameras["third"].set_fovx(self.standard_head_cam_fovx, compute_y=True)
            # self.cameras["wrist"].set_fovx(self.standard_wrist_cam_fovx, compute_y=True)

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
            if not self.door_from_urdf:
                if self.use_real:
                    init_p = self.np_random.uniform(
                        low=np.array([-0.6, 0.4, 0.0]), high=np.array([-0.5, 0.4, 0.0])
                    )
                    init_angle = self.np_random.uniform(low=-0.1, high=0.1)
                else:
                    init_p = self.np_random.uniform(
                        low=np.array([-0.6, 0.0, 0.0]), high=np.array([-0.5, 0.0, 0.0])
                    )
                    # init_p = np.array([-0.5, 0.0, 0.0])
                    init_angle = self.np_random.uniform(
                        low=-np.pi / 2 - 0.01, high=-np.pi / 2 + 0.01
                    )
            else:
                init_p = self.np_random.uniform(
                    low=np.array([-1.2, 0.0, 0.0]), high=np.array([-1.1, 0.0, 0.0])
                )
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

        self.door_articulation.set_qpos(np.zeros((self.door_articulation.dof,)))
        self.init_agv_pose = self._get_agv_pose()
        # print("In reset, init_agv_pose", self.init_agv_pose)
        self.init_base_pose = self._get_base_pose()
        # reset stage for expert policy
        self.expert_phase = 0

        # TODO: get obs
        self._update_observation()
        obs = OrderedDict()
        for key in self.observation_dict.keys():
            obs[key] = self.observation_dict[key]
        return obs, {}

    def step(self, action: np.ndarray):
        action = action.copy()
        info = {}

        if len(action) == 7:
            # action is desired base w, desired base v, delta eef pose, binary gripper action in the specified frame
            cur_ee_pose = self._get_tcp_pose()
            cur_relative_ee_pose = self.init_agv_pose.inv().transform(cur_ee_pose)
            # action relative to tool frame
            if self.action_relative == "tool":
                desired_relative_ee_pose = cur_relative_ee_pose.transform(
                    sapien.Pose(
                        p=action[0:3],
                        q=euler2quat(action[3:6]),
                    )
                )
            # action relative to fixed frame
            elif self.action_relative == "base":
                desired_relative_ee_pose = sapien.Pose(
                    cur_relative_ee_pose.p + self.p_scale * action[0:3],
                    euler2quat(
                        quat2euler(cur_relative_ee_pose.q)
                        + self.rot_scale * action[3:6]
                    ),
                )
            # pose in initial agv frame
            elif self.action_relative == "none":
                desired_relative_ee_pose = sapien.Pose(
                    p=action[0:3], q=euler2quat(action[3:6])
                )
                # dirty
                # action[8] = -action[8]
            desired_ee_pose = self.init_agv_pose.transform(desired_relative_ee_pose)

            target_qpos = self.base_arm_controller.compute_q_target(
                 desired_ee_pose, 0.85 - action[6]  ## np.zeros(2),
            )
            info["desired_relative_pose"] = np.concatenate(
                [desired_relative_ee_pose.p, desired_relative_ee_pose.q]
            )
        else:
            assert len(action) == 8
            target_arm_q = action[:7]
            target_gripper_q = (0.85 - action[7]) * np.ones(2)  
            target_qpos = np.concatenate(
                [ target_arm_q, target_gripper_q]  ## np.zeros(2),
            )
        info["desired_joints"] = target_qpos[
            self.base_arm_controller.arm_joint_indices
        ]
        info["desired_gripper_width"] = (
            0.85 - target_qpos[self.base_arm_controller.finger_joint_indices[0]]
        )
        # target_qvel = np.concatenate(
        #     [action[:2] * self.vel_scale, np.zeros(self.robot.dof - 2)]
        # )

        self.robot.set_drive_target(target_qpos)
        # self.robot.set_drive_velocity_target(target_qvel)
        self.robot.set_qf(
            self.robot.compute_passive_force(
                external=False, coriolis_and_centrifugal=False
            )
        )
        handle_joint = self.door_articulation.get_active_joints()[1]
        handle_joint.set_drive_target(0.0)
        for i in range(self.frame_skip):
            self.scene.step()

        # TODO: obs, reward, info
        self._update_observation()
        obs = OrderedDict()
        for key in self.observation_dict.keys():
            obs[key] = self.observation_dict[key]
        is_success = self._is_success()
        reward = (
            5. * self._reward_door_angle()
            + 5. * self._reward_handle_angle()
            + 5. * self._reward_approach_handle()
            + 1. * is_success
        )  # TODO
        # print("In step, init_agv_pose", self.init_agv_pose)
        info.update(
            {
                "is_success": is_success,
                "init_agv_pose": self.init_agv_pose,
                "reward": reward,
            }
        )
        return obs, reward, is_success, False, info

    def expert_action(self, noise_scale=0.0):
        # phases: before pregrasp, to grasp, close gripper, rotate, pull open
        handle_T_grasp = sapien.Pose.from_transformation_matrix(
            np.array(
                [
                    [0, -np.sin(np.pi / 6 * 0), np.cos(np.pi / 6 * 0), 0.02],
                    [-1, 0, 0, -0.05],
                    [0, -np.cos(np.pi / 6 * 0), -np.sin(np.pi / 6 * 0), 0],
                    [0, 0, 0, 1],
                ]
            )
        )
        handle_pose = self._get_handle_pose()
        # print(handle_pose, "handle_pose")  # for debug
        desired_grasp_pose: sapien.Pose = None

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
            handle_T_pregrasp = sapien.Pose(
                p=np.array([-0.1, 0.0, 0.0]), q=handle_T_grasp.q
            )
            # print(handle_T_pregrasp, "handle_T_pregrasp")  # for debug
            desired_grasp_pose = handle_pose.transform(handle_T_pregrasp)
            # print(desired_grasp_pose, "desired_grasp_pose")  # for debug
            apply_noise_to_pose(desired_grasp_pose)
            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                0.85 + self.np_random.uniform(-0.02, 0.02) * noise_scale,
            )
            # print(action, "action")   # for debug
        elif self.expert_phase == 1:
            desired_grasp_pose = handle_pose.transform(handle_T_grasp)
            apply_noise_to_pose(desired_grasp_pose)
            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                0.85 + self.np_random.uniform(-0.02, 0.02) * noise_scale,
            )
        elif self.expert_phase == 2:
            gripper_width = self._get_gripper_width()
            desired_grasp_pose = handle_pose.transform(handle_T_grasp)
            apply_noise_to_pose(desired_grasp_pose)
            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                gripper_width - 0.2 + self.np_random.uniform(-0.02, 0.02) * noise_scale,
            )
        elif self.expert_phase == 3:
            handle_joint_pose = self._get_handle_joint_pose()
            rotation = sapien.Pose(
                p=np.zeros(3), q=euler2quat(np.array([np.pi / 20, 0, 0]))
            )
            hjoint_T_handle = handle_joint_pose.inv().transform(handle_pose)
            desired_grasp_pose = (
                handle_joint_pose.transform(rotation)
                .transform(hjoint_T_handle)
                .transform(handle_T_grasp)
            )
            apply_noise_to_pose(desired_grasp_pose)
            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                0.0 + self.np_random.uniform(-0.02, 0.02) * noise_scale,
            )
            # print("phase3 desired pose", desired_grasp_pose.to_transformation_matrix(),
            #       "cur tcp pose", self._get_tcp_pose().to_transformation_matrix())
            # print("handle_joint_pose", handle_joint_pose.to_transformation_matrix(),
            #       "hjoint_T_handle", hjoint_T_handle.to_transformation_matrix(),
            #       )
        elif self.expert_phase == 4:
            door_joint_pose = self._get_door_joint_pose()
            rotation = sapien.Pose(
                p=np.zeros(3), q=euler2quat(np.array([np.pi / 90, 0, 0]))
            )
            djoint_T_grasp = door_joint_pose.inv().transform(self._get_tcp_pose())
            desired_grasp_pose = door_joint_pose.transform(rotation).transform(
                djoint_T_grasp
            )
            apply_noise_to_pose(desired_grasp_pose)
            action = self._desired_tcp_to_action(
                desired_grasp_pose,
                0.0 + self.np_random.uniform(-0.02, 0.02) * noise_scale,
            )
            # print("phase4 desired pose", desired_grasp_pose.to_transformation_matrix(), "cur pose", self._get_tcp_pose().to_transformation_matrix())
            # print("door joint pose", door_joint_pose.to_transformation_matrix(),
            #       "d joint T grasp", djoint_T_grasp.to_transformation_matrix())
            # exit()
        else:
            raise NotImplementedError
        # TODO: error recovery
        tcp_pose = self._get_tcp_pose()
        if self.expert_phase == 0:
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
            if self._is_grasp(both_finger=True):
                # print("door joint q", self.door_articulation.get_qpos()[0])    # for debug
                if (
                    self.door_articulation.get_qpos()[0] < 0.1 and self.need_door_shut
                ):  # door is closed, need to rotate
                    print("switch to phase3")
                    self.expert_phase = 3
                else:
                    print("switch to phase4")
                    self.expert_phase = 4  # door is open, only need to pull
        elif self.expert_phase == 3:
            if self.door_articulation.get_qpos()[1] > np.pi / 5:
                self.expert_phase = 4
        elif self.expert_phase == 4:
            # pass
            if not self._is_grasp():  # lost grasp, need to regrasp
                self.expert_phase = 0
        return action

    # compute all the observations
    def _update_observation(self):
        self.observation_dict.clear()
        if self.use_image_obs:
            image_obs = self.capture_images_new()
        world_tcp_pose = self._get_tcp_pose()
        tcp_pose = self.init_agv_pose.inv().transform(self._get_tcp_pose())
        gripper_width = self._get_gripper_width()
        door_states = self.door_articulation.get_qpos()
        handle_pose = self._get_handle_pose()
        arm_joints = self.robot.get_qpos()[self.base_arm_controller.arm_joint_indices]
        if self.use_image_obs:
            self.observation_dict.update(image_obs)
        self.observation_dict["tcp_pose"] = np.concatenate([tcp_pose.p, tcp_pose.q])
        self.observation_dict["gripper_width"] = gripper_width
        self.observation_dict["robot_joints"] = arm_joints
        self.observation_dict["door_states"] = door_states
        self.observation_dict["privileged_obs"] = np.concatenate(
            [
                world_tcp_pose.p,
                world_tcp_pose.q,
                [gripper_width],
                handle_pose.p,
                handle_pose.q,
                door_states,
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

    def _get_base_pose(self) -> sapien.Pose:
        return self.robot.get_pose()

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
            # base_action = np.array([base_angv, base_linv]) / self.vel_scale
            return np.concatenate(
                [
                    # base_action,
                    delta_pos,
                    delta_euler,
                    [gripper_width],  # [(0.85 - gripper_width) / 0.85 * 2 - 1],
                ]
            )

    def _is_grasp(self, threshold: float = 1e-4, both_finger=False):
        all_contact = self.scene.get_contacts()
        robot_finger_links: List[sapien.LinkBase] = [
            self.robot.get_links()[i] for i in self.finger_link_idxs
        ]
        door_handle_link = self.door_articulation.get_links()[self.door_handle_link_idx]
        if not self.door_from_urdf:
            door_handler_cs = door_handle_link.get_collision_shapes()[1]
        else:
            door_handler_cs = door_handle_link.get_collision_shapes()[0]
        finger_impulses = [
            get_pairwise_contact_impulse(
                all_contact, robot_finger, door_handle_link, None, door_handler_cs
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


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_filename = "test_open_door"
    video_writer = imageio.get_writer(
        f"{video_filename}.mp4",
        fps=40,
        format="FFMPEG",
        codec="h264",
    )
    open_door_sim = OpenDoorEnv(
        use_gui=False,
        device=device,
        obs_keys=("third-rgb", "tcp_pose", "gripper_width"),
        door_from_urdf=False,
        # use_real=True,
        # domain_randomize=False,
    )
    open_door_sim.reset()
    step_count = 0
    traj_count = 0
    success_count = 0
    done = False
    while traj_count < 10:
        action = open_door_sim.expert_action(noise_scale=0.5)
        # exit()
        _, _, _, _, info = open_door_sim.step(action)
        step_count += 1
        rgb_image = open_door_sim.render()
        video_writer.append_data(rgb_image)
        done = info["is_success"] or step_count >= 600
        if done:
            traj_count += 1
            step_count = 0
            success_count += info["is_success"]
            open_door_sim.reset()
    print("success rate", success_count / traj_count)
    video_writer.close()


if __name__ == "__main__":
    test()
