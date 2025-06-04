import gym
import numpy as np
from gym import spaces
from sapien.core import Pose
from collections import OrderedDict, deque
from .multi_step import repeated_space
import torch
import pickle
from transforms3d.quaternions import quat2mat
from transforms3d.euler import mat2euler
from env.sapien_utils.math import get_pose_from_rot_pos
import torchvision.transforms as transforms

class SapienPickAndPlaceWrapper(gym.Wrapper):
    """
        SAPIEN Pick and Place Real rl wrapper
        support venv reset and step    
    """
    
    def __init__(
            self, 
            env, 
            record=False,
            n_obs_steps=4, 
            n_action_steps=4, 
            max_episode_steps=200, 
            normalization_path=None,
            asynchronous=True,
        ):

        super().__init__(env)
        self.seed_value = None
        self.record = record
        self.normalization_path = normalization_path
        self.asynchronous = asynchronous

        # obs is include state(10-dim, include gripper width) and image
        self._single_observation_space = spaces.Dict({
            "state": spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32), 
            "rgb": spaces.Box(low=0., high=1.0, shape=(3, 224, 224), dtype=np.float32),  
        })

        self._single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        
        if normalization_path is not None:
            OBS_NORMALIZE_PARAMS = pickle.load(open(normalization_path, "rb"))
            self.pose_gripper_mean = np.concatenate(
                [
                    OBS_NORMALIZE_PARAMS[key]["mean"]
                    for key in ["pose", "gripper_width"]
                ]
            )
            self.pose_gripper_scale = np.concatenate(
                [
                    OBS_NORMALIZE_PARAMS[key]["scale"]
                    for key in ["pose", "gripper_width"]
                ]
            )
            self.proprio_gripper_mean = np.concatenate(
                [
                    OBS_NORMALIZE_PARAMS[key]["mean"]
                    for key in ["proprio_state", "gripper_width"]
                ]
            )
            self.proprio_gripper_scale = np.concatenate(
                [
                    OBS_NORMALIZE_PARAMS[key]["scale"]
                    for key in ["proprio_state", "gripper_width"]
                ]
            )

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        self.env_steps = 0
        
        self.action_space = repeated_space(self._single_action_space, n_action_steps)
        self.observation_space = repeated_space(self._single_observation_space, n_obs_steps)
        
        self.obs = deque(maxlen=n_obs_steps+1)
        
        if hasattr(self.env, 'max_episode_steps'):
            self.env.max_episode_steps = self.max_episode_steps
    
    def process_observation(self, obs):
        """
            Args:
                obs: dict, observation from env
            Returns:
                processed_obs: dict, processed observation
        """
        # process observation
        single_obs = {}
        pose = obs["tcp_pose"]
        pose_p, pose_q = pose[:3], pose[3:]
        pose_mat = quat2mat(pose_q)
        pose_mat_6 = pose_mat[:, :2].reshape(-1)
        single_obs["state"] = (
            np.concatenate([
                pose_p,
                pose_mat_6,
                np.array([obs["gripper_width"]]),
            ]) - self.proprio_gripper_mean) / self.proprio_gripper_scale

        images = obs["third-rgb"]
        assert len(images.shape) == 3,  f"Warning: Unexpected image shape for third-rgb: {images.shape}"

        original_size = images.shape[:2]
        ratio = 0.95
        transformations = [
            transforms.CenterCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
            transforms.Resize((224, 224), antialias=True),
        ]
        
        images_tensor = torch.from_numpy(images).float()
        images_tensor = torch.einsum('h w c -> c h w', images_tensor)
        
        for transform in transformations:
            images_tensor = transform(images_tensor)
            
        # [C, H, W]
        images_tensor = images_tensor / 255.0
        single_obs["rgb"] = images_tensor

        return single_obs

    def reset(self, options_list=None):
        if self.seed_value is not None:
            seed = self.seed_value 
        else:
            seed = None
        
        self.obs.clear()
        obs, info = self.env.reset(seed=seed, options=options_list)
        self.env_steps = 0

        # process observation
        single_obs = self.process_observation(obs) 
        self.obs.append(single_obs)
        stacked_obs = self._stack_last_n_obs_dict(self.obs, self.n_obs_steps)

        return stacked_obs, info

    def seed(self, seed=None):
        """
        Args:
            seed: int
        """
        self.seed_value = seed 

    def reset_arg(self, options_list=None):
        return self.reset(options_list=options_list) 

    def reset_one_arg(self, env_ind=None, options=None):
        """
            not in use yet.
        """
        if env_ind is not None:
            env_ind = torch.tensor([env_ind], device=self.device)
        return self.reset()

    def step(self, action: np.ndarray):
        """
            step multi-step action,
            include 10-dim -> 7-dim, and unormalize
            
            Args:
                action: [n_envs, n_action_steps, 10], 
                        normalized action chunk, n_action_steps is the actual execution steps
        """
        total_reward = 0
        terminated = False
        truncated = False
        info = {}
        success_array =[]
        record_list = []
        # only use state for pose_at_obs
        obs = self.env.get_observation(use_image=False)
        pose = obs["tcp_pose"]
        pose_p, pose_q = pose[:3], pose[3:]
        pose_mat = quat2mat(pose_q)
        pose_at_obs = get_pose_from_rot_pos(pose_mat, pose_p)
        
        if self.asynchronous == False:
            assert len(action.shape) == 3, f"Expected action shape 3, [1, n_steps, action_dim], got {action.shape}"
            action = action[0, :, :]
        else:
            assert len(action.shape) == 2, f"Expected action shape 2, for single env in asynchronous envs, got {action.shape}"

        # unnormalize
        if self.normalization_path is not None:
            axis = (0,1) if len(action.shape) == 3 else (0,)
            action = action * np.expand_dims(self.pose_gripper_scale, axis=axis) \
                            + np.expand_dims(self.pose_gripper_mean, axis=axis)
       
        n_steps_to_execute = self.n_action_steps
        converted_actions = []

        for i in range(n_steps_to_execute):
            current_action = action[i] # [10]

            mat_6 = current_action[3:9].reshape(3, 2)
            mat_6[:, 0] = mat_6[:, 0] / np.linalg.norm(mat_6[:, 0])
            mat_6[:, 1] = mat_6[:, 1] / np.linalg.norm(mat_6[:, 1])
            z_vec = np.cross(mat_6[:, 0], mat_6[:, 1])
            mat = np.c_[mat_6, z_vec]
            
            pos = current_action[:3]
            gripper_width = current_action[-1]
            
            init_to_desired_pose = pose_at_obs @ get_pose_from_rot_pos(mat, pos)
            
            # trans to 7d
            pose_action = np.concatenate([
                init_to_desired_pose[:3, 3],  #
                mat2euler(init_to_desired_pose[:3, :3]),  # 3d
                [gripper_width]  
            ])
            converted_actions.append(pose_action)

            obs, step_reward, step_terminated, step_truncated, step_info = self.env.step(pose_action)
            
            if self.record:
              record_list.append(obs["third-rgb"])
            
            single_obs = self.process_observation(obs)
            self.obs.append(single_obs)
            if len(self.obs) > self.n_obs_steps:
                self.obs.popleft()
                
            total_reward += step_reward
            terminated = terminated or step_terminated
            truncated = truncated or step_truncated
            
            if "is_success" in step_info:
                success_array.append(float(step_info["is_success"]))
        
        # [n_action_steps]
        info["is_success"] = np.array(success_array)

        stacked_obs = self._stack_last_n_obs_dict(self.obs, self.n_obs_steps)
        self.env_steps += n_steps_to_execute
        
        if self.max_episode_steps is not None and self.env_steps >= self.max_episode_steps:
            truncated = True
            terminated = True
            info["TimeLimit.truncated"] = True
            
        if terminated or truncated:
            self.env_steps = 0
            self.reset()
        
        if self.record:
            info["record_list"] = record_list

        info["converted_actions"] = converted_actions
        
        return stacked_obs, total_reward, terminated, truncated, info
    
    def _stack_last_n_obs_dict(self, all_obs, n_steps):
        """
            Args:
                all_obs: queue of dicts, each dict contains the observation
                n_steps: int, the number of steps to stack
            Returns:
                stacked_obs: 
                    asynchronous envs:
                        {key: (n_obs_steps, *obs_dims)}
                    synchronous envs:
                        {key: (n_envs=1, n_obs_steps, *obs_dims)}
        """
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        n_envs = 1  # only support one env yet
    
        recent_obs = []
        
        if len(all_obs) < n_steps:
            padding = [all_obs[0]] * (n_steps - len(all_obs))
            recent_obs = padding + all_obs
        else:
            recent_obs = all_obs[-n_steps:]
        
        result = {}
        for key in recent_obs[0].keys():
            key_obs_list = [obs[key] for obs in recent_obs]
            # (n_envs, n_obs_steps, ...)
            if isinstance(key_obs_list[0], torch.Tensor):
                key_obs_list = [tensor.cpu().numpy() if tensor.requires_grad else tensor.numpy() 
                         for tensor in key_obs_list]
            if self.asynchronous == False:
                key_obs_array = np.expand_dims(np.stack(key_obs_list), axis=0)
            else:
                key_obs_array = np.stack(key_obs_list)
            result[key] = key_obs_array
        
        return result
        
    def _get_observation(self):
        return self.env.get_observation()

    ##### subproc venv ######

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.capture_images()

    def get_obs(self):
        return self._get_observation()