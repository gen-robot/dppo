import numpy as np
import os
import sapien.core as sapien
import torchvision
from PIL import Image
from typing import List
from env.sapien_utils.math import wrap_to_pi, euler2quat, quat2euler
# from homebot_sapien import PANDA_DATA
PANDA_DATA = os.path.expanduser("~/panda_data")


dtd_cache: List[str] = []


class DRConfig:
    dtd_root: str


def sample_random_texture(dtd_root: str, np_random: np.random.RandomState):
    global dtd_cache
    if len(dtd_cache) == 0:
        torchvision.datasets.DTD(root=dtd_root, download=True, split="train")
        dtd_cache = []
        for split in ["train", "val", "test"]:
            dtd_cache.extend(
                [
                    path.strip()
                    for path in open(
                        os.path.join(dtd_root, f"dtd/dtd/labels/{split}1.txt"), "r"
                    ).readlines()
                ]
            )

        dtd_cache = sorted(set(dtd_cache))
    jpg_path = os.path.join(dtd_root, "dtd/dtd/images/", np_random.choice(dtd_cache))
    return jpg_path

def apply_cano_texture(
    material: sapien.RenderMaterial
):
    file_path = os.path.join(PANDA_DATA, "texture", "pure", "proce_white.jpg")
    if isinstance(material, List):
        for m in material:
            m.set_diffuse_texture_from_file(file_path)
    else:
        material.set_diffuse_texture_from_file(file_path)


def apply_random_texture(
    material: sapien.RenderMaterial, np_random: np.random.RandomState
):
    # dtd_root = "/root/data/texture"
    dtd_root = save_dir = os.path.join(PANDA_DATA, "texture")
    randomized_file_path = sample_random_texture(dtd_root, np_random)
    # Check which is/are useful
    # material.set_base_color(np.concatenate([[1.0, 0.0, 0.0], [1.0]]))
    if isinstance(material, List):
        for m in material:
            m.set_diffuse_texture_from_file(randomized_file_path)
    else:
        # print(randomized_file_path)
        # material.set_base_color(np.concatenate([[0.0, 0.0, 0.0], [0.0]]))
        material.set_diffuse_texture_from_file(randomized_file_path)
        # material.set_normal_texture_from_file(randomized_file_path)
        # material.set_emission_texture_from_file(randomized_file_path)
        # material.set_metallic_texture_from_file(randomized_file_path)


def check_intersect_2d(p1, bbox1, p2, bbox2):
    intersect_x = max(p1[0] + bbox1["min"][0], p2[0] + bbox2["min"][0]) < min(p1[0] + bbox1["max"][0], p2[0] + bbox2["max"][0])
    intersect_y = max(p1[1] + bbox1["min"][1], p2[1] + bbox2["min"][1]) < min(p1[1] + bbox1["max"][1], p2[1] + bbox2["max"][1])
    return intersect_x and intersect_y


def check_intersect_2d_(p1, bbox1, scale1, p2, bbox2, scale2):
    r_1 = np.sqrt(bbox1["max"][0] ** 2 + bbox1["max"][1] ** 2) * scale1
    r_2 = np.sqrt(bbox2["max"][0] ** 2 + bbox2["max"][1] ** 2) * scale2
    dist_12 = np.sqrt(np.sum((p1[:2] - p2[:2]) ** 2))
    return r_1 + r_2 > dist_12


def grasp_pose_process(grasp_pose):
    grasp_euler = quat2euler(grasp_pose.q)
    center = np.pi / 4 * 1
    half_range = np.pi / 8 * 5
    # center = 0
    if abs(grasp_euler[-1] - center) > half_range:
        grasp_euler[-1] -= np.sign(grasp_euler[-1] - center) * np.pi
    return sapien.Pose(p=grasp_pose.p, q=euler2quat(grasp_euler))


