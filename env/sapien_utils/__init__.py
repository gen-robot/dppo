import torch
from env.homebot.pick_and_place_panda_real_rl import PickAndPlaceEnv
from env.homebot.microwave import MicrowavePushAndPullEnv
from env.homebot.open_door import OpenDoorEnv
from env.homebot.drawer import DrawerPushAndPullEnv

from env.sapien_utils.homebot_multistep_wrapper import HomeBotMultiStepWrapper
from env.sapien_utils.subproc_vec_env import SubprocVecEnv


HOMEBOT_ENV_DICT = {
    "pick_and_place": PickAndPlaceEnv, 
    "microwave": MicrowavePushAndPullEnv, 
    "open_door": OpenDoorEnv, 
    "drawer": DrawerPushAndPullEnv
}

def make_async_sapien(
    id,
    num_envs=1,
    asynchronous=True,
    wrappers=None,
    use_gui=False,
    obs_dim=23,
    action_dim=7,
    env_type=None,
    max_episode_steps=None,
    # below for furniture only
    gpu_id=0,
    headless=True,
    record=False,
    normalization_path=None,
    obs_steps=1,
    act_steps=8,
    sparse_reward=False,
    # below for robomimic only
    use_image_obs=False,
    render_offscreen=False,
    shape_meta=None,
    **kwargs,
):
    if env_type == "sapien":
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env_cls = HOMEBOT_ENV_DICT[id]
        
        if asynchronous:
            def make_env(idx):
                def _init():
                    env_seed = kwargs.get('seed', 0) + idx if 'seed' in kwargs else None
                    
                    env = env_cls(
                        use_gui=(use_gui and idx == 0), # only record in the first environment
                        device=device,
                        obs_keys=("tcp_pose", "gripper_width", "privileged_obs", "third-rgb"),
                        use_image_obs=use_image_obs,
                        action_relative="none",
                        domain_randomize=True,
                        canonical=True
                    )
                    
                    env = HomeBotMultiStepWrapper(
                        env,
                        n_obs_steps=obs_steps,
                        n_action_steps=act_steps,
                        max_episode_steps=max_episode_steps,
                        normalization_path=normalization_path,
                        asynchronous=asynchronous,
                    )
                    return env
                return _init
            
            env_fns = [make_env(i) for i in range(num_envs)]
            venv = SubprocVecEnv(env_fns, start_method='spawn')
            return venv
        else:
            env = env_cls(
                use_gui=use_gui,
                device=device,
                obs_keys=("tcp_pose", "gripper_width", "privileged_obs", "third-rgb"),
                use_image_obs=use_image_obs,
                action_relative="none",
                domain_randomize=True,
                canonical=True
            )   
            env = SapienPickAndPlaceWrapper(
                env,
                n_obs_steps=obs_steps,
                n_action_steps=act_steps,
                max_episode_steps=max_episode_steps,
                normalization_path=normalization_path,
            )
            return env