import gym
import numpy as np
from gym import spaces
from sapien.core import Pose
from collections import OrderedDict, deque
from .multi_step import repeated_space

class SapienPickAndPlaceMultiStepWrapper(gym.Wrapper):
    """SAPIEN Pick and Place环境的Wrapper类,支持多步动作"""
    
    def __init__(self, env, n_obs_steps=4, n_action_steps=4, max_episode_steps=200, render=False):
        """初始化wrapper
        
        Args:
            env: SAPIEN环境实例
            n_obs_steps: 观察步数
            n_action_steps: 动作步数
            max_episode_steps: 最大回合步数，默认200
            render: 是否渲染环境
        """
        super().__init__(env)
        
        # 设置观察空间
        self._single_observation_space = spaces.Dict({
            "tcp_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),  # TCP位姿 (位置3 + 四元数4)
            "gripper_width": spaces.Box(low=0, high=0.1, shape=(1,), dtype=np.float32),  # 夹爪宽度
            "robot_joints": spaces.Box(low=-np.pi, high=np.pi, shape=(7,), dtype=np.float32),  # 机器人关节位置
            "privileged_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),  # 特权观察 (世界坐标系下的TCP位姿 + 夹爪宽度)
        })
        
        # 设置动作空间 (TCP位置增量 + 夹爪宽度)
        self._single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        
        # 多步设置
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        self.env_steps = 0
        
        # 扩展动作空间以支持多步
        self.action_space = repeated_space(self._single_action_space, n_action_steps)
        
        # 扩展观察空间以支持多步
        self.observation_space = repeated_space(self._single_observation_space, n_obs_steps)
        
        # 初始化控制器参数
        self.p_scale = 0.05  # 位置缩放
        self.rot_scale = 0.1  # 旋转缩放
        self.gripper_scale = 0.01  # 夹爪缩放
        self.gripper_limit = 0.08  # 夹爪限制
        
        # 初始化观察队列
        self.obs = deque(maxlen=max(n_obs_steps + 1, n_action_steps))
        
        # 设置环境的max_episode_steps
        if hasattr(self.env, 'max_episode_steps'):
            self.env.max_episode_steps = self.max_episode_steps
        
    def reset(self, seed=None, options=None):
        """重置环境
        
        Args:
            seed: 随机种子
            options: 其他选项
            
        Returns:
            observation: 初始观察
            info: 额外信息
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.env_steps = 0
        self.obs = deque([obs], maxlen=max(self.n_obs_steps + 1, self.n_action_steps))
        stacked_obs = self._stack_last_n_obs_dict(self.obs, self.n_obs_steps)
        
        # 打印初始状态
        #print("Environment reset:")
        #print(f"Initial TCP pose: {self.env._get_tcp_pose().p}")
        #print(f"Initial object pose: {self.env.objs[list(self.env.objs.keys())[0]]['actor'].get_pose().p}")
        #print(f"Initial gripper width: {self.env._get_gripper_width()}")
        
        return stacked_obs, info
        
    def step(self, action: np.ndarray):
        """执行多步动作
        
        Args:
            action: 动作向量 [n_envs, n_action_steps, 7]
            
        Returns:
            observation: 新的观察
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        # 执行n_action_steps步
        total_reward = 0
        terminated = False
        truncated = False
        info = {}
        
        # 确保动作形状正确
        if len(action.shape) == 3:
            action = action[0]  # 取第一个环境的动作
        
        # 确保不超过实际的动作步数
        actual_steps = min(self.n_action_steps, action.shape[0])
        
        # 记录初始状态
        initial_tcp_pose = self.env._get_tcp_pose()
        initial_obj_pose = self.env.objs[list(self.env.objs.keys())[0]]["actor"].get_pose()
        
        for i in range(actual_steps):
            # 获取当前步骤的动作，确保是一维数组
            current_action = action[i].flatten()  # 形状为[7]
            
            # 执行单步动作
            obs, reward, is_success, step_terminated, step_truncated, step_info = self.env.step(current_action)
            
            # 更新观察队列
            self.obs.append(obs)
            if len(self.obs) > self.n_obs_steps:
                self.obs.popleft()
                
            # 使用compute_normalized_dense_reward计算归一化奖励
            step_reward = self.env.compute_normalized_dense_reward(obs, current_action, step_info)
            total_reward += step_reward
            
            # 检查是否终止
            terminated = terminated or step_terminated
            truncated = truncated or step_truncated
            
            # 计算success_rate
            obj_pose = self.env.objs[list(self.env.objs.keys())[0]]["actor"].get_pose()
            goal_pos = np.array([0.4, -0.2, self.env.table_top_z])
            obj_to_goal_dist = np.linalg.norm(obj_pose.p - goal_pos)
            is_success = obj_to_goal_dist < 0.1 and self.env._is_grasp(self.env.objs[list(self.env.objs.keys())[0]]["actor"], both_finger=True)
            success_rate = float(is_success)
            
            # 合并信息
            for k, v in step_info.items():
                if k not in info:
                    info[k] = v
                elif isinstance(v, np.ndarray):
                    info[k] = np.concatenate([info[k], v])
                else:
                    info[k] = v
            info["success_rate"] = success_rate
            
            # 如果终止或截断，则结束当前步骤
            if terminated or truncated:
                break
        
        # 计算总位移
        final_tcp_pose = self.env._get_tcp_pose()
        final_obj_pose = self.env.objs[list(self.env.objs.keys())[0]]["actor"].get_pose()
        tcp_displacement = np.linalg.norm(final_tcp_pose.p - initial_tcp_pose.p)
        obj_displacement = np.linalg.norm(final_obj_pose.p - initial_obj_pose.p)
        
        # 堆叠最后n_obs_steps步的观察
        stacked_obs = self._stack_last_n_obs_dict(self.obs, self.n_obs_steps)
        
        # 更新环境步数
        self.env_steps += actual_steps
        
        # 检查是否达到最大步数
        if self.max_episode_steps is not None and self.env_steps >= self.max_episode_steps:
            truncated = True
            terminated = True
            info["TimeLimit.truncated"] = True
            
        # 如果回合终止或截断，重置环境
        if terminated or truncated:
            self.env_steps = 0
            self.reset()
            
        #print(f"\nDisplacement - TCP: {tcp_displacement:.4f}, Object: {obj_displacement:.4f}")
        #print(f"Total reward: {total_reward}")
        #print(f"Terminated: {terminated}, Truncated: {truncated}")
        
        return stacked_obs, total_reward, terminated, truncated, info
    
    def _stack_last_n_obs_dict(self, all_obs, n_steps):
        """堆叠最后n步的观察
        
        Args:
            all_obs: 所有观察的列表
            n_steps: 要堆叠的步数
            
        Returns:
            stacked_obs: 堆叠后的观察字典
        """
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        
        # 创建结果字典
        result = {}
        for key in all_obs[-1]:
            # 获取观察的形状
            shape = all_obs[-1][key].shape
            # 创建堆叠后的数组
            stacked = np.zeros((n_steps,) + shape, dtype=all_obs[-1][key].dtype)
            
            # 填充堆叠数组
            start_idx = -min(n_steps, len(all_obs))
            for i, obs in enumerate(all_obs[start_idx:]):
                stacked[i + n_steps - len(all_obs[start_idx:])] = obs[key]
            
            # 如果步数不足，用第一个观察填充
            if n_steps > len(all_obs):
                for i in range(n_steps - len(all_obs)):
                    stacked[i] = stacked[n_steps - len(all_obs)]
            
            result[key] = stacked
        
        return result
        
    def _get_observation(self):
        """获取当前观察
        
        Returns:
            observation: 观察字典
        """
        # 获取TCP位姿
        world_tcp_pose = self.env._get_tcp_pose()
        tcp_pose = self.env.init_base_pose.inv().transform(world_tcp_pose)
        
        # 获取夹爪宽度
        gripper_width = self.env._get_gripper_width()
        
        # 获取机器人关节位置
        arm_joints = self.env.robot.get_qpos()[self.env.arm_controller.arm_joint_indices]
        
        # 构建观察字典
        obs = OrderedDict()
        obs["tcp_pose"] = np.concatenate([tcp_pose.p, tcp_pose.q])
        obs["gripper_width"] = np.array([gripper_width])
        obs["robot_joints"] = arm_joints
        obs["privileged_obs"] = np.concatenate([
            world_tcp_pose.p,
            world_tcp_pose.q,
            [gripper_width]
        ])
        
        return obs
        
    def _compute_reward(self):
        """计算奖励
        
        Returns:
            reward: 奖励值
        """
        # 获取物体和目标位姿
        obj_pose = self.env.objs[list(self.env.objs.keys())[0]]["actor"].get_pose()
        goal_pos = np.array([0.4, -0.2, self.env.table_top_z])
        
        # 计算TCP到物体的距离
        tcp_pose = self.env._get_tcp_pose()
        tcp_to_obj_dist = np.linalg.norm(obj_pose.p - tcp_pose.p)
        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)  # 增加缩放因子
        
        # 检查是否抓取成功
        is_grasped = self.env._is_grasp(self.env.objs[list(self.env.objs.keys())[0]]["actor"], both_finger=True)
        grasp_reward = 2.0 * float(is_grasped)
        
        # 计算物体到目标位置的距离
        obj_to_goal_dist = np.linalg.norm(goal_pos - obj_pose.p)
        place_reward = 1 - np.tanh(5 * obj_to_goal_dist)  # 增加缩放因子
        
        # 计算静态奖励
        qvel = self.env.robot.get_qvel()
        qvel = qvel[:-2]  # 排除夹爪关节
        static_reward = 1 - np.tanh(5 * np.linalg.norm(qvel))  # 增加缩放因子
        
        # 组合奖励
        reward = reaching_reward + grasp_reward + place_reward + static_reward
        
        # 检查是否成功放置
        is_success = obj_to_goal_dist < 0.1 and is_grasped
        if is_success:
            reward = 10.0
            
        return reward