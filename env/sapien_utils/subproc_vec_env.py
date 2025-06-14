import multiprocessing
from collections import OrderedDict
from typing import Sequence

import gymnasium as gym
import numpy as np
import os

from .base_vec_env import VecEnv, CloudpickleWrapper


def _worker(remote, parent_remote, env_fn_wrapper, reset_when_done):
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # os.environ["OMP_NUM_THREADS"] = "1"

    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, truncated, info = env.step(data)
                if done or truncated:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    if reset_when_done:
                        observation, _ = env.reset()
                remote.send((observation, reward, done, truncated, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation, info = env.reset()
                remote.send((observation, info))
            elif cmd == "reset_arg": # add
                observation, info = env.reset(**data)
                remote.send((observation, info))
            elif cmd == "get_obs":
                observation = env.get_obs()
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: ([callable]) A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, start_method=None, reset_when_done=True):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = (
                "forkserver" in multiprocessing.get_all_start_methods()
            )
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe(duplex=True) for _ in range(n_envs)]
        )
        self.processes = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn), reset_when_done)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, truncates, infos = zip(*results)
        return (
            _flatten_obs(obs, self.observation_space),
            np.stack(rews),
            np.stack(dones),
            np.stack(truncates),
            _flatten_info(infos),
        )

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset_arg(self, **kwargs):
        for remote in self.remotes:
            remote.send(("reset_arg", kwargs))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), _flatten_info(infos)

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), _flatten_info(infos)

    # def get_obs(self, indices=None):
    #     target_remotes = self._get_target_remotes(indices)
    #     for remote in target_remotes:
    #         remote.send(('get_obs', None))
    #     obs = [remote.recv() for remote in self.remotes]
    #     return _flatten_obs(obs, self.observation_space)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self, *args, **kwargs) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            # pipe.send(('render', (args, {'mode': 'rgb_array', **kwargs})))
            pipe.send(("render", None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def dispatch_env_method(
        self, method_name, *dispatched_args, n_args=1, indices=None, **method_kwargs
    ):
        target_remotes = self._get_target_remotes(indices)
        for idx, remote in enumerate(target_remotes):
            # print("in subproc", type(dispatched_args[idx]))
            # print(dispatched_args[idx].shape)
            # print(type(dispatched_args[idx][0]))
            if n_args == 1:
                method_args = [dispatched_args[idx]]
            else:
                method_args = [dispatched_args[j][idx] for j in range(n_args)]
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return (OrderedDict<ndarray>, tuple<ndarray> or ndarray) flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(
        obs, (list, tuple)
    ), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"


    # print(f"Space type: {type(space)}, name: {type(space).__name__}")
    # print(f"Is Dict: {isinstance(space, gym.spaces.Dict)}")
    # print(f"Is OrderedDict: {isinstance(space, OrderedDict)}")

    is_dict_space = (
        isinstance(space, gym.spaces.Dict) or 
        type(space).__name__ == 'Dict' or
        hasattr(space, 'spaces')
    )
    
    if is_dict_space and hasattr(space, 'spaces'):
        assert isinstance(
            obs[0], dict
        ), "non-dict observation for environment with Dict observation space"
        
        return OrderedDict(
            [(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()]
        )
    # if isinstance(space, gym.spaces.Dict):
    #     print("--------------orderdict--------------") ##
    #     assert isinstance(
    #         space.spaces, OrderedDict
    #     ), "Dict space must have ordered subspaces"
    #     assert isinstance(
    #         obs[0], dict
    #     ), "non-dict observation for environment with Dict observation space"
        
    #     return OrderedDict(
    #         [(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()]
    #     )
    elif isinstance(space, gym.spaces.Tuple):
        print("------------tuple") ##
        assert isinstance(
            obs[0], tuple
        ), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        print("--------stack-----------") ##
        return np.stack(obs)


def _flatten_info(infos):
    """
    将多个环境的 info 字典堆叠在一起
    
    :param infos: (list<dict>) 每个环境的 info 字典列表
    :return: (dict) 堆叠后的 info 字典，每个值都是一个数组，第一个维度是环境索引
    """
    assert isinstance(infos, (list, tuple)), "expected list or tuple of info dicts"
    assert len(infos) > 0, "need info from at least one environment"
    
    keys = set()
    for info in infos:
        keys.update(info.keys())
    
    stacked_infos = {}
    for key in keys:
        values = [info.get(key) for info in infos]
        try:
            stacked_infos[key] = np.stack(values)
        except:
            stacked_infos[key] = values
    
    return stacked_infos