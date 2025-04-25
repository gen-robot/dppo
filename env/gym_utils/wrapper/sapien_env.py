import gym
import numpy as np
from gym import spaces
import sapien.core as sapien
from sapien.utils import Viewer
import imageio
from collections import OrderedDict
import os
import torch
from transforms3d.quaternions import qmult, qconjugate, quat2mat, mat2quat
from transforms3d.euler import euler2quat, quat2euler

class SapienEnvWrapper(gym.Env):
    """SAPIEN环境的基础wrapper类，适配PickAndPlaceEnv"""
    
    def __init__(self, env, render=False, render_offscreen=False, render_hw=(480, 640), obs_keys=None, 
                 action_relative="tool", domain_randomize=True, canonical=True, allow_dir=None):
        super().__init__()
        self.env = env
        self.render_mode = render
        self.render_offscreen = render_offscreen
        self.render_hw = render_hw
        self.obs_keys = obs_keys or []
        self.action_relative = action_relative
        self.domain_randomize = domain_randomize
        self.canonical = canonical
        self.allow_dir = allow_dir or []
        
        # 设置观察空间
        self.observation_space = spaces.Dict()
        
        # 添加图像观察空间
        if "third-rgb" in self.obs_keys:
            self.observation_space["third-rgb"] = spaces.Box(
                low=0, high=255, shape=(240, 320, 3), dtype=np.uint8
            )
            
        # 添加TCP姿态观察空间
        if "tcp_pose" in self.obs_keys:
            self.observation_space["tcp_pose"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )
            
        # 添加夹爪宽度观察空间
        if "gripper_width" in self.obs_keys:
            self.observation_space["gripper_width"] = spaces.Box(
                low=0, high=0.04, shape=(1,), dtype=np.float32
            )
            
        # 添加机器人关节观察空间
        if "robot_joints" in self.obs_keys:
            self.observation_space["robot_joints"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )
            
        # 添加特权观察空间
        if "privileged_obs" in self.obs_keys:
            self.observation_space["privileged_obs"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
            )
        
        # 设置动作空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )
        
        # 初始化渲染器
        if render or render_offscreen:
            self.viewer = Viewer(self.env.renderer)
            self.viewer.set_camera_xyz(0, -3, 1)
            self.viewer.set_camera_rpy(0, -0.5, 0)
        else:
            self.viewer = None
            
    def reset(self, options={}, **kwargs):
        """重置环境"""
        seed = options.get("seed", None)
        obj_list = options.get("obj_list", None)
        obs, info = self.env.reset(seed=seed, options=options, obj_list=obj_list)
        return self._process_obs(obs)
        
    def step(self, action):
        """执行动作"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info
        
    def _process_obs(self, obs):
        """处理观察值"""
        if isinstance(obs, dict):
            return obs
        return {"state": obs}
        
    def render(self, mode="rgb_array"):
        """渲染环境"""
        if self.viewer is None:
            return None
            
        if mode == "rgb_array":
            self.viewer.render()
            return self.viewer.get_camera_images()["Color"]
        elif mode == "human":
            self.viewer.render()
            return None
            
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
        self.env.close()
        
    def seed(self, seed=None):
        """设置随机种子"""
        self.env.seed(seed)
        return [seed]
        
    def get_observation(self, use_image=True):
        """获取观察值"""
        return self.env.get_observation(use_image=use_image)
        
    def expert_action(self, obj_id, goal_obj_pose, noise_scale=0.0):
        """获取专家动作"""
        return self.env.expert_action(obj_id, goal_obj_pose, noise_scale)
        
    def reload_objs(self, obj_list=None, egad_ratio=0.5, num_obj=1):
        """重新加载物体"""
        return self.env.reload_objs(obj_list=obj_list, egad_ratio=egad_ratio, num_obj=num_obj)
        
    def _get_tcp_pose(self):
        """获取TCP姿态"""
        return self.env._get_tcp_pose()
        
    def _get_gripper_width(self):
        """获取夹爪宽度"""
        return self.env._get_gripper_width()
        
    def _get_base_pose(self):
        """获取基座姿态"""
        return self.env._get_base_pose()
        
    def _is_grasp(self, actor, threshold=1e-4, both_finger=False):
        """检查是否抓取成功"""
        return self.env._is_grasp(actor, threshold, both_finger)
        
    def capture_images_new(self):
        """捕获图像"""
        return self.env.capture_images_new()
        
    def save_images(self, save_dir):
        """保存图像"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        images = self.capture_images_new()
        for name, img in images.items():
            imageio.imwrite(os.path.join(save_dir, f"{name}.jpg"), img)
            
    def test(self):
        """测试环境"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 测试环境初始化
        obs = self.get_observation()
        print("初始观察:", obs.keys())
        
        # 测试动作执行
        action = np.array([0.3, 0, 0.3, np.pi, 0, 0, 0.04])
        obs, reward, terminated, truncated, info = self.step(action)
        print("执行动作后的观察:", obs.keys())
        
        # 保存图像
        self.save_images("tmp")
        print("图像已保存到tmp目录")
        
        return obs, reward, terminated, truncated, info 