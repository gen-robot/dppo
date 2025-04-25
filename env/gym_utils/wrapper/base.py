import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer
from typing import Dict, List, Optional, Tuple, Union, Sequence

MAX_DEPTH_RANGE = 2.5


class BaseEnv(gym.Env):
    def __init__(
        self,
        use_gui: bool,
        device: str,
        mipmap_levels=1,
    ):
        super().__init__()
        self.engine = sapien.Engine()

        # sapien.render_config.camera_shader_dir = "rt"
        sapien.render_config.camera_shader_dir = "ibl"
        # sapien.render_config.viewer_shader_dir = "rt"
        # sapien.render_config.rt_samples_per_pixel = 64 * 4
        # sapien.render_config.rt_use_denoiser = False

        self.renderer = sapien.SapienRenderer(
            default_mipmap_levels=mipmap_levels,
            offscreen_only=not use_gui,
            device='cuda',
            do_not_load_texture=False,
            max_num_materials=5000,
            max_num_textures=5000,
            culling='back',
        )
        self.engine.set_renderer(self.renderer)
        # sapien.VulkanRenderer.set_camera_shader_dir("ibl")
        if use_gui:
            # sapien.VulkanRenderer.set_viewer_shader_dir("ibl")
            viewer = Viewer(self.renderer)
            viewer.close()
        self.engine.set_log_level("error")

        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.005)

        # Dummy camera creation to initial geometry object
        if self.renderer:
            cam = self.scene.add_camera(
                "init_not_used", width=10, height=10, fovy=1., near=0.1, far=1
            )
            self.scene.remove_camera(cam)
            print("add and remove camera")

        self.cameras: Dict[str, sapien.CameraEntity] = {}
        self.frame_skip = 10

        self.load_scene()

    def seed(self, seed: int):
        self._np_random, _ = seeding.np_random(seed)

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed, options=options)

    def step(self):
        raise NotImplementedError

    def load_scene(self):
        pass

    def create_camera(
        self,
        position: np.ndarray,
        look_at_dir: np.ndarray,
        right_dir: np.ndarray,
        name: str,
        resolution: Sequence[Union[float, int]],
        fov: float,
        mount_actor: sapien.ActorBase = None,
    ):
        if not len(resolution) == 2:
            raise ValueError(
                f"Resolution should be a 2d array, but now {len(resolution)} is given."
            )
        if mount_actor is not None:
            # for actor in self.scene.get_all_actors():
            #     print("actor name", actor.get_name())
            # mount = [actor for actor in self.scene.get_all_actors() if actor.get_name() == mount_actor_name]
            mount = [mount_actor]
            # if len(mount) == 0:
            #     raise ValueError(f"Camera mount {mount_actor_name} not found in the env.")
            # if len(mount) > 1:
            #     raise ValueError(
            #         f"Camera mount {mount_actor_name} name duplicates! To mount an camera on an actor,"
            #         f" give the mount a unique name.")
            mount = mount[0]
            cam = self.scene.add_mounted_camera(
                name,
                mount,
                sapien.Pose(),
                width=resolution[0],
                height=resolution[1],
                fovy=fov,
                fovx=fov,
                near=0.01,
                far=10,
            )
        else:
            # Construct camera pose
            look_at_dir = look_at_dir / np.linalg.norm(look_at_dir)
            right_dir = (
                right_dir
                - np.sum(right_dir * look_at_dir).astype(np.float64) * look_at_dir
            )
            right_dir = right_dir / np.linalg.norm(right_dir)
            up_dir = np.cross(look_at_dir, -right_dir)
            rot_mat_homo = np.stack([look_at_dir, -right_dir, up_dir, position], axis=1)
            pose_mat = np.concatenate([rot_mat_homo, np.array([[0, 0, 0, 1]])])
            pose_cam = sapien.Pose.from_transformation_matrix(pose_mat)
            cam = self.scene.add_camera(
                name,
                width=resolution[0],
                height=resolution[1],
                fovy=fov,
                near=0.1,
                far=10,
            )
            cam.set_local_pose(pose_cam)

        self.cameras.update({name: cam})

    def capture_images(self):
        self.scene.update_render()
        obs_dict = {}
        for name in self.cameras:
            cam = self.cameras[name]
            modalities = ["rgb", "depth", "segmentation"]
            texture_names = []
            for modality in modalities:
                if modality == "rgb":
                    texture_names.append("Color")
                elif modality == "depth":
                    texture_names.append("Position")
                elif modality == "point_cloud":
                    texture_names.append("Position")
                elif modality == "segmentation":
                    texture_names.append("Segmentation")
                else:
                    raise ValueError(f"Visual modality {modality} not supported.")

            await_dl_list = cam.take_picture_and_get_dl_tensors_async(texture_names)
            dl_list = await_dl_list.wait()

            for i, modality in enumerate(modalities):
                key_name = f"{name}-{modality}"
                dl_tensor = dl_list[i]
                shape = sapien.dlpack.dl_shape(dl_tensor)
                if modality in ["segmentation"]:
                    # TODO: add uint8 async
                    import torch

                    output_array = torch.from_dlpack(dl_tensor).cpu().numpy()
                else:
                    output_array = np.zeros(shape, dtype=np.float32)
                    sapien.dlpack.dl_to_numpy_cuda_async_unchecked(
                        dl_tensor, output_array
                    )
                    sapien.dlpack.dl_cuda_sync()
                if modality == "rgb":
                    obs = output_array[..., :3]
                    obs = np.clip(obs * 255, 0, 255).astype(np.uint8)
                elif modality == "depth":
                    obs = -output_array[..., 2:3]
                    obs[
                        obs[..., 0] > MAX_DEPTH_RANGE
                    ] = 0  # Set depth out of range to be 0
                # elif modality == "point_cloud":
                #     obs = np.reshape(output_array[..., :3], (-1, 3))
                #     camera_pose = self.get_camera_to_robot_pose(name)
                #     kwargs = camera_cfg["point_cloud"].get("process_fn_kwargs", {})
                #     obs = camera_cfg["point_cloud"]["process_fn"](obs, camera_pose,
                #                                                   camera_cfg["point_cloud"]["num_points"],
                #                                                   self.np_random, **kwargs)
                #     if "additional_process_fn" in camera_cfg["point_cloud"]:
                #         for fn in camera_cfg["point_cloud"]["additional_process_fn"]:
                #             obs = fn(obs, self.np_random)
                elif modality == "segmentation":
                    obs = output_array[..., :2].astype(np.uint8)
                else:
                    raise RuntimeError("What happen? you should not see this error!")
                obs_dict[key_name] = obs

        # if len(self.imaginations) > 0:
        #     obs_dict.update(self.imaginations)

        return obs_dict

    def capture_images_new(self, cameras=None):
        self.scene.update_render()
        obs_dict = {}
        if cameras is None:
            cameras = self.cameras
        for name in cameras:
            self.cameras[name].take_picture()
            # modalities = ["rgb", "depth"]
            modalities = ["rgb"]
            for modality in modalities:
                key_name = f"{name}-{modality}"
                if modality == "rgb":
                    rgba = self.cameras[name].get_float_texture("Color")  # [H, W, 4]
                    obs = np.clip(rgba[..., :3] * 255, 0, 255).astype(np.uint8)
                elif modality == "depth":
                    # Each pixel is (x, y, z, render_depth) in camera space (OpenGL/Blender)
                    position = self.cameras[name].get_float_texture(
                        "Position"
                    )  # [H, W, 4]
                    depth = -position[..., 2]
                    obs = (depth * 1000.0).astype(np.uint16)
                else:
                    raise NotImplementedError
                obs_dict[key_name] = obs
        return obs_dict

    def render(self):
        image_dict = self.capture_images_new()
        return image_dict["third-rgb"]

    def render_all(self):
        image_dict = self.capture_images()
        return image_dict


def recover_action(action, limit):
    action = (action + 1) / 2 * (limit[:, 1] - limit[:, 0]) + limit[:, 0]
    return action


def set_render_material(material: sapien.RenderMaterial, **kwargs):
    for k, v in kwargs.items():
        if k == "color":
            material.set_base_color(v)
        else:
            setattr(material, k, v)
    return material


def set_articulation_render_material(articulation: sapien.Articulation, **kwargs):
    for link in articulation.get_links():
        for b in link.get_visual_bodies():
            for s in b.get_render_shapes():
                mat = s.material
                set_render_material(mat, **kwargs)


def get_pairwise_contacts(
    contacts: List[sapien.Contact],
    actor0: sapien.ActorBase,
    actor1: sapien.ActorBase,
    collision_shape0: Optional[sapien.CollisionShape] = None,
    collision_shape1: Optional[sapien.CollisionShape] = None,
) -> List[Tuple[sapien.Contact, bool]]:
    pairwise_contacts = []
    for contact in contacts:
        if (
            contact.actor0 == actor0
            and contact.actor1 == actor1
            and (
                collision_shape0 is None or contact.collision_shape0 == collision_shape0
            )
            and (
                collision_shape1 is None or contact.collision_shape1 == collision_shape1
            )
        ):
            pairwise_contacts.append((contact, True))
        elif (
            contact.actor0 == actor1
            and contact.actor1 == actor0
            and (
                collision_shape1 is None or contact.collision_shape0 == collision_shape1
            )
            and (
                collision_shape0 is None or contact.collision_shape1 == collision_shape0
            )
        ):
            pairwise_contacts.append((contact, False))
    return pairwise_contacts


def compute_total_impulse(contact_infos: List[Tuple[sapien.Contact, bool]]):
    total_impulse = np.zeros(3)
    for contact, flag in contact_infos:
        contact_impulse = np.sum([point.impulse for point in contact.points], axis=0)
        # Impulse is applied on the first actor
        total_impulse += contact_impulse * (1 if flag else -1)
    return total_impulse


def get_pairwise_contact_impulse(
    contacts: List[sapien.Contact],
    actor0: sapien.ActorBase,
    actor1: sapien.ActorBase,
    collision_shape0: Optional[sapien.CollisionShape] = None,
    collision_shape1: Optional[sapien.CollisionShape] = None,
):
    pairwise_contacts = get_pairwise_contacts(
        contacts, actor0, actor1, collision_shape0, collision_shape1
    )
    total_impulse = compute_total_impulse(pairwise_contacts)
    return total_impulse
