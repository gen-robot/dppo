import numpy as np
import os
import sapien.core as sapien
from enum import IntEnum


class ArmControlMode(IntEnum):
    EEF_POS = 0


def load_robot_full(scene: sapien.Scene, arm_control_mode=ArmControlMode.EEF_POS):
    finger_link_idxs = []
    loader = scene.create_urdf_loader()
    filename = os.path.join(
        os.path.dirname(__file__), "../../asset/xarm7_with_gripper.urdf"
    )
    robot_builder = loader.load_file_as_articulation_builder(filename)
    for link_builder in robot_builder.get_link_builders():
        link_builder.set_collision_groups(1, 1, 17, 0)
    # if disable_self_collision:
    #     for link_builder in robot_builder.get_link_builders():
    #         link_builder.set_collision_groups(1, 1, 17, 0)
    # else:
    #     if "allegro" in robot_name:
    #         for link_builder in robot_builder.get_link_builders():
    #             if link_builder.get_name() in ["link_9.0", "link_5.0", "link_1.0", "link_13.0", "base_link"]:
    #                 link_builder.set_collision_groups(1, 1, 17, 0)
    robot = robot_builder.build(fix_root_link=True)
    robot_name = "homebot"
    robot.set_name(robot_name)
    for link_idx, link in enumerate(robot.get_links()):
        link_name = link.get_name()
        if link_name == "right_finger" or link_name == "left_finger":
            finger_link_idxs.append(link_idx)
        if "finger" in link_name or "knuckle" in link_name:
            vb_list = link.get_visual_bodies()
            for vb in vb_list:
                for rs in vb.get_render_shapes():
                    material = rs.material
                    # black gripper
                    material.set_base_color(np.array([0.0, 0.0, 0.0, 1.0]))
                    rs.set_material(material)

    # robot_arm_control_params = np.array([0, 300, 300])
    # robot_arm_control_params = np.array([200000, 40000, 500])  # This PD is far larger than real to improve stability
    if arm_control_mode == ArmControlMode.EEF_POS:
        robot_arm_control_params = np.array([1000, 100, 100])
    # elif arm_control_mode == ArmControlMode.EEF_VEL:
    #     robot_arm_control_params = np.array([0.0, 300, 300])
    # Stiffness, damping, force_limit
    root_translation_control_params = np.array([0, 1000, 100000])
    # root_translation_control_params = np.array([20, 0.0, 20])
    root_rotation_control_params = np.array([0, 1000, 100000])
    finger_control_params = np.array([10000, 1000, 10000])

    if False:
        # if "free" in robot_name:
        for joint in robot.get_active_joints():
            name = joint.get_name()
            print(name)
            if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                print("drive property", root_translation_control_params)
                joint.set_drive_property(
                    *(1 * root_translation_control_params), mode="force"
                )
            elif (
                    "x_rotation_joint" in name
                    or "y_rotation_joint" in name
                    or "z_rotation_joint" in name
            ):
                print("drive property", root_rotation_control_params)
                joint.set_drive_property(
                    *(1 * root_rotation_control_params), mode="force"
                )
            else:
                print("drive property", finger_control_params)
                joint.set_drive_property(*(1 * finger_control_params), mode="force")
    else:
        mobile_rotation_name = ["mobile_rotation"]
        mobile_translation_name = ["mobile_translation"]
        arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        for joint in robot.get_active_joints():
            name = joint.get_name()
            if name in mobile_rotation_name:
                joint.set_drive_property(*root_rotation_control_params, mode="force")
                joint.set_friction(0.01)
            elif name in mobile_translation_name:
                joint.set_drive_property(*root_translation_control_params, mode="force")
                joint.set_friction(0.01)
            elif name in arm_joint_names:
                joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
            else:
                joint.set_drive_property(*(1 * finger_control_params), mode="force")

    mat = scene.engine.create_physical_material(1.5, 1, 0.01)
    rough_mat = scene.engine.create_physical_material(5, 5, 0.0)
    for link in robot.get_links():
        if "finger" in link.get_name():
            for geom in link.get_collision_shapes():
                geom.min_patch_radius = 0.1
                geom.patch_radius = 0.1
                geom.set_physical_material(rough_mat)
        else:
            for geom in link.get_collision_shapes():
                geom.min_patch_radius = 0.02
                geom.patch_radius = 0.04
                geom.set_physical_material(mat)
    return robot, finger_link_idxs


def load_robot_panda(scene: sapien.Scene, arm_control_mode=ArmControlMode.EEF_POS):
    finger_link_idxs = []
    loader = scene.create_urdf_loader()
    filename = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "asset",
        "franka_panda",
        "panda.urdf"
    )
    robot_builder = loader.load_file_as_articulation_builder(filename)

    # for link_builder in robot_builder.get_link_builders():
    #     link_builder.set_collision_groups(1, 1, 17, 0)
    # if disable_self_collision:
    #     for link_builder in robot_builder.get_link_builders():
    #         link_builder.set_collision_groups(1, 1, 17, 0)
    # else:
    #     if "allegro" in robot_name:
    #         for link_builder in robot_builder.get_link_builders():
    #             if link_builder.get_name() in ["link_9.0", "link_5.0", "link_1.0", "link_13.0", "base_link"]:
    #                 link_builder.set_collision_groups(1, 1, 17, 0)
    robot = robot_builder.build(fix_root_link=True)
    robot_name = "panda"
    robot.set_name(robot_name)
    for link_idx, link in enumerate(robot.get_links()):
        link_name = link.get_name()
        if link_name == "panda_leftfinger" or link_name == "panda_rightfinger":
            finger_link_idxs.append(link_idx)
        elif link_name == "panda_link0":
            vb_list = link.get_visual_bodies()
            for vb in vb_list:
                for rs in vb.get_render_shapes():
                    material = rs.material
                    material.set_base_color(np.array([1.0, 1.0, 1.0, 1.0]))
        # if "finger" in link_name or "knuckle" in link_name:
        #     vb_list = link.get_visual_bodies()
        #     for vb in vb_list:
        #         for rs in vb.get_render_shapes():
        #             material = rs.material
        #             # black gripper
        #             material.set_base_color(np.array([0.0, 0.0, 0.0, 1.0]))
        #             rs.set_material(material)

    # robot_arm_control_params = np.array([0, 300, 300])
    # robot_arm_control_params = np.array([200000, 40000, 500])  # This PD is far larger than real to improve stability
    if arm_control_mode == ArmControlMode.EEF_POS:
        robot_arm_control_params = np.array([1000, 100, 100])
    # elif arm_control_mode == ArmControlMode.EEF_VEL:
    #     robot_arm_control_params = np.array([0.0, 300, 300])
    # Stiffness, damping, force_limit
    # finger_control_params = np.array([10000, 1000, 10000])
    finger_control_params = np.array([1000, 100, 10000])   #zzt

    arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]
    for joint in robot.get_active_joints():
        name = joint.get_name()
        if name in arm_joint_names:
            joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
        else:
            joint.set_drive_property(*(1 * finger_control_params), mode="force")

    mat = scene.engine.create_physical_material(1.5, 1, 0.01)
    # rough_mat = scene.engine.create_physical_material(5, 5, 0.0)
    rough_mat = scene.engine.create_physical_material(10, 10, 0.0)
    for link in robot.get_links():
        if "finger" in link.get_name():
            for geom in link.get_collision_shapes():
                geom.min_patch_radius = 0.1
                geom.patch_radius = 0.1
                geom.set_physical_material(rough_mat)
        else:
            for geom in link.get_collision_shapes():
                geom.min_patch_radius = 0.02
                geom.patch_radius = 0.04
                geom.set_physical_material(mat)
    return robot, finger_link_idxs

