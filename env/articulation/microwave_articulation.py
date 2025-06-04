import dataclasses
import numpy as np
import os
import sapien.core as sapien
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from typing import List

# from pathlib import Path
ASSET_DIR = os.path.join(os.path.dirname(__file__), "../../asset")


def load_microwave_urdf(scene: sapien.Scene, id=7320, scale=1.0):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    loader.scale = scale
    urdf_path = os.path.join(
        ASSET_DIR, f"partnet_mobility/{id}/mobility.urdf" 
    )
    microwave_articulation = loader.load(
        urdf_path,
    )
    # for link in door_articulation.get_links():
    #     if link.get_name() == "link_switch_and_lock":
    #         for visualbodies in link.get_visual_bodies():
    #             if visualbodies.get_name() == "lock_bar":
    #                 lock_bar_pose_p = visualbodies.local_pose.p
    #                 break
    #         for s in link.get_collision_shapes():
    #             if (lock_bar_pose_p == s.get_local_pose().p).all():
    #                 s.set_collision_groups(2, 2, 0, 0)
    #                 lock_bar_pose_p = 0
    #             else:
    #                 s.set_collision_groups(3, 3, 1 << 31, 0)
    #     else:
    #         for s in link.get_collision_shapes():
    #             g0, g1, g2, g3 = s.get_collision_groups()
    #             s.set_collision_groups(3, 3, 1 << 31, 0)
    # for test
    # for link in door_articulation.get_links():
    #     print("linkname:",link.get_name())
    #     for s in link.get_collision_shapes():
    #         g0, g1, g2, g3 = s.get_collision_groups()
    #         print("door after:g0,g1,g2,g3",hex(g0),hex(g1), hex(g2), hex(g3))

    microwave_joint = microwave_articulation.get_active_joints()[0]
    microwave_joint.set_drive_property(0, 1, 5)
    # door_joint.set_drive_property(0, 1, 5)
    # handle_joint.set_drive_property(1, 0, 10)

    return microwave_articulation


def load_microwaves(scene: sapien.Scene, scale: float, poses: List[sapien.Pose]):
    num_microwaves = len(poses)
    microwaves = [load_microwave_urdf(scene, scale=scale) for _ in range(num_microwaves)]
    for microwave, pose in zip(microwaves, poses):
        microwave.set_root_pose(pose)

    return microwaves


def load_table_4(
        scene: sapien.Scene,
        surface_size=np.array([0.8, 1.0, 0.03]),
        leg_size=np.array([0.04, 0.04, 0.73]),
        leg_pos_x=0.3,
        leg_pos_y=0.4,
        root_position=np.array([0.23, 0.0, 0.0]),
        root_angle=0
):
    """
    The origin is at the middle of the bottom of the four legs
    """
    table_builder = scene.create_actor_builder()
    # legs

    leg_pos_z = leg_size[2] / 2
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([-leg_pos_x, -leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([-leg_pos_x, -leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([-leg_pos_x, leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([-leg_pos_x, leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([leg_pos_x, -leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([leg_pos_x, -leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([leg_pos_x, leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([leg_pos_x, leg_pos_y, leg_pos_z])),
        half_size=leg_size / 2,
    )
    # upper surface
    # surface_size = np.array([1.0, 1.6, 0.05])
    surface_pos_x = 0.0
    surface_pos_y = 0.0
    surface_pos_z = leg_size[2] + surface_size[2] / 2
    table_builder.add_box_collision(
        sapien.Pose(p=np.array([surface_pos_x, surface_pos_y, surface_pos_z])),
        half_size=surface_size / 2,
    )
    table_builder.add_box_visual(
        sapien.Pose(p=np.array([surface_pos_x, surface_pos_y, surface_pos_z])),
        half_size=surface_size / 2,
        color=np.array([0., 0.4, 0.]),
        name="upper_surface"
    )
    table = table_builder.build(name="table")
    table.set_pose(
        sapien.Pose(
            root_position,
            np.array(
                [np.cos(root_angle / 2), 0.0, 0.0, np.sin(root_angle / 2)]  # (w, x, y, z)
            )
        )
    )
    table.lock_motion()
    return table