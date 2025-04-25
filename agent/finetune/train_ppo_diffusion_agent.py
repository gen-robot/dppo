"""
DPPO fine-tuning.

"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import math
import collections
import time

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_agent import TrainPPOAgent
from util.scheduler import CosineAnnealingWarmupRestarts


class TrainPPODiffusionAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Reward horizon --- always set to act_steps for now
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)

        # Eta - between DDIM (=0 for eval) and DDPM (=1 for training)
        self.learn_eta = self.model.learn_eta
        if self.learn_eta:
            self.eta_update_interval = cfg.train.eta_update_interval
            self.eta_optimizer = torch.optim.AdamW(
                self.model.eta.parameters(),
                lr=cfg.train.eta_lr,
                weight_decay=cfg.train.eta_weight_decay,
            )
            self.eta_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.eta_optimizer,
                first_cycle_steps=cfg.train.eta_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.eta_lr,
                min_lr=cfg.train.eta_lr_scheduler.min_lr,
                warmup_steps=cfg.train.eta_lr_scheduler.warmup_steps,
                gamma=1.0,
            )

    def get_state(self, obs):
        if isinstance(obs, tuple):
            obs_dict = obs[0]
        else:
            obs_dict = obs

        if isinstance(obs_dict, collections.OrderedDict) or isinstance(obs_dict, dict):
            if "state" in obs_dict:
                state = np.array(obs_dict["state"])
                # 确保状态是2D数组 [batch_size, state_dim]
                if len(state.shape) == 1:
                    state = state.reshape(1, -1)
                return state
            else:
                # 如果没有state键，将所有观察值连接起来
                state_values = []
                for key, value in obs_dict.items():
                    if isinstance(value, np.ndarray):
                        state_values.append(value.flatten())
                    elif np.isscalar(value):
                        state_values.append(np.array([value]))
                    else:
                        try:
                            value_array = np.array(value)
                            state_values.append(value_array.flatten())
                        except:
                            print(f"Warning: Could not convert {key} to numpy array")
                state = np.concatenate(state_values)
                # 确保状态是2D数组 [batch_size, state_dim]
                if len(state.shape) == 1:
                    state = state.reshape(1, -1)
                return state
        else:
            # 如果不是字典类型，尝试直接转换为numpy数组
            try:
                state = np.array(obs_dict)
                # 确保状态是2D数组 [batch_size, state_dim]
                if len(state.shape) == 1:
                    state = state.reshape(1, -1)
                return state
            except:
                print("Warning: Could not convert observation to numpy array")
                # 返回一个零向量作为默认值，确保维度正确
                return np.zeros((1, self.obs_dim))

    def run(self):
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:
            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv

            # Holder
            obs_trajs = {
                "state": np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim)
                )
            }
            chains_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            if self.save_full_observations:  # state-only
                obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, self.get_state(prev_obs_venv)[:, -1][None])
                )

            # Collect a set of trajectories from env
            env_step_time = 0.0
            network_update_time = 0.0
            
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch.no_grad():
                    # 处理SAPIEN环境的观察值
                    if isinstance(prev_obs_venv, tuple):
                        # 如果是元组，第一个元素是观察值，第二个元素是信息
                        obs_dict = prev_obs_venv[0]
                        if isinstance(obs_dict, collections.OrderedDict):
                            # 如果是OrderedDict，将所有观察值连接起来
                            state_values = []
                            for key, value in obs_dict.items():
                                if isinstance(value, np.ndarray):
                                    state_values.append(value.flatten())
                                elif np.isscalar(value):
                                    state_values.append(np.array([value]))
                                else:
                                    # 处理其他类型的值
                                    try:
                                        value_array = np.array(value)
                                        state_values.append(value_array.flatten())
                                    except:
                                        print(f"Warning: Could not convert {key} to numpy array")
                            obs_trajs["state"][step] = np.concatenate(state_values)
                        else:
                            # 如果不是OrderedDict，直接使用第一个元素
                            obs_trajs["state"][step] = obs_dict
                    elif isinstance(prev_obs_venv, (collections.OrderedDict, dict)):
                        # 如果是OrderedDict或dict类型，需要提取状态值
                        if "state" in prev_obs_venv:
                            # 确保state是numpy数组
                            state = prev_obs_venv["state"]
                            if not isinstance(state, np.ndarray):
                                state = np.array(state)
                            obs_trajs["state"][step] = state
                        else:
                            # 如果没有state键，尝试将所有观察值连接起来
                            state_values = []
                            for key, value in prev_obs_venv.items():
                                if isinstance(value, np.ndarray):
                                    state_values.append(value.flatten())
                                elif np.isscalar(value):
                                    state_values.append(np.array([value]))
                                else:
                                    # 处理其他类型的值
                                    try:
                                        value_array = np.array(value)
                                        state_values.append(value_array.flatten())
                                    except:
                                        print(f"Warning: Could not convert {key} to numpy array")
                            obs_trajs["state"][step] = np.concatenate(state_values)
                    else:
                        # 处理其他类型的观察值
                        try:
                            obs_trajs["state"][step] = self.get_state(prev_obs_venv)
                        except:
                            print(f"Warning: Could not convert observation to numpy array")
                            # 提供一个默认值
                            obs_trajs["state"][step] = np.zeros(self.obs_dim)
                    cond = {
                        "state": torch.from_numpy(obs_trajs["state"][step]).float().to(self.device)
                    }
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        return_chain=True,
                    )
                    output_venv = (
                        samples.trajectories.cpu().numpy()
                    )  # n_env x horizon x act
                    chains_venv = (
                        samples.chains.cpu().numpy()
                    )  # n_env x denoising x horizon x act
                action_venv = output_venv[:, : self.act_steps]

                # Apply multi-step action
                import time
                step_start = time.time()
                (
                    obs_venv,
                    reward_venv,
                    terminated_venv,
                    truncated_venv,
                    info_venv,
                ) = self.venv.step(action_venv)
                env_step_time += time.time() - step_start
                done_venv = terminated_venv | truncated_venv
                
                # 每50步打印一次调试信息
                #if step % 50 == 0:
                    #print(f"\nStep {step}:")
                    #print(f"Action shape: {action_venv.shape}")
                    #print(f"Raw Reward: {reward_venv}")
                    #print(f"Reward scale const: {self.reward_scale_const}")
                    #print(f"Scaled Reward: {reward_venv * self.reward_scale_const}")
                    #print(f"Terminated: {terminated_venv}")
                    #print(f"Truncated: {truncated_venv}")
                    #print(f"Done: {done_venv}")
                    #print(f"Firsts trajs: {firsts_trajs[step]}")
                
                if self.save_full_observations:  # state-only
                    obs_full_venv = np.array(
                        [info["full_obs"]["state"] for info in info_venv]
                    )  # n_envs x act_steps x obs_dim
                    obs_full_trajs = np.vstack(
                        (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                    )
                # 更新观察值存储
                if isinstance(prev_obs_venv, tuple):
                    # 如果是元组，第一个元素是观察值，第二个元素是信息
                    obs_dict = prev_obs_venv[0]
                    if isinstance(obs_dict, collections.OrderedDict):
                        # 如果是OrderedDict，将所有观察值连接起来
                        state_values = []
                        for key, value in obs_dict.items():
                            if isinstance(value, np.ndarray):
                                state_values.append(value.flatten())
                            elif np.isscalar(value):
                                state_values.append(np.array([value]))
                            else:
                                # 处理其他类型的值
                                try:
                                    value_array = np.array(value)
                                    state_values.append(value_array.flatten())
                                except:
                                    print(f"Warning: Could not convert {key} to numpy array")
                        obs_trajs["state"][step] = np.concatenate(state_values)
                    else:
                        # 如果不是OrderedDict，直接使用第一个元素
                        obs_trajs["state"][step] = obs_dict
                elif isinstance(prev_obs_venv, (collections.OrderedDict, dict)):
                    # 如果是OrderedDict或dict类型，需要提取状态值
                    if "state" in prev_obs_venv:
                        # 确保state是numpy数组
                        state = prev_obs_venv["state"]
                        if not isinstance(state, np.ndarray):
                            state = np.array(state)
                        obs_trajs["state"][step] = state
                    else:
                        # 如果没有state键，尝试将所有观察值连接起来
                        state_values = []
                        for key, value in prev_obs_venv.items():
                            if isinstance(value, np.ndarray):
                                state_values.append(value.flatten())
                            elif np.isscalar(value):
                                state_values.append(np.array([value]))
                            else:
                                # 处理其他类型的值
                                try:
                                    value_array = np.array(value)
                                    state_values.append(value_array.flatten())
                                except:
                                    print(f"Warning: Could not convert {key} to numpy array")
                        obs_trajs["state"][step] = np.concatenate(state_values)
                else:
                    # 处理其他类型的观察值
                    try:
                        obs_trajs["state"][step] = self.get_state(prev_obs_venv)
                    except:
                        print(f"Warning: Could not convert observation to numpy array")
                        # 提供一个默认值
                        obs_trajs["state"][step] = np.zeros(self.obs_dim)
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv

                # update for next step
                prev_obs_venv = obs_venv

                # count steps --- not acounting for done within action chunk
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            # 计算episode的开始和结束
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                log.info(f"Environment {env_ind} first steps: {env_steps}")
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
                        log.info(f"Found episode: env={env_ind}, start={start}, end={end-1}")
            
            log.info(f"Total episodes found: {len(episodes_start_end)}")
            log.info(f"Firsts trajs shape: {firsts_trajs.shape}")
            log.info(f"Firsts trajs sum: {np.sum(firsts_trajs)}")
            
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                if (
                    self.furniture_sparse_reward
                ):  # only for furniture tasks, where reward only occurs in one env step
                    episode_best_reward = episode_reward
                else:
                    episode_best_reward = np.array(
                        [
                            np.max(reward_traj) / self.act_steps
                            for reward_traj in reward_trajs_split
                        ]
                    )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = info_venv.get("success_rate", 0.0)
            else:
                num_episode_finished = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # Update models
            if not eval_mode:
                update_start = time.time()
                with torch.no_grad():
                    obs_trajs["state"] = (
                        torch.from_numpy(obs_trajs["state"]).float().to(self.device)
                    )

                    # Calculate value and logprobs - split into batches to prevent out of memory
                    num_split = math.ceil(
                        self.n_envs * self.n_steps / self.logprob_batch_size
                    )
                    obs_ts = [{} for _ in range(num_split)]
                    obs_k = einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                    obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                    for i, obs_t in enumerate(obs_ts_k):
                        obs_ts[i]["state"] = obs_t
                    values_trajs = np.empty((0, self.n_envs))
                    for obs in obs_ts:
                        values = self.model.critic(obs).cpu().numpy().flatten()
                        values_trajs = np.vstack(
                            (values_trajs, values.reshape(-1, self.n_envs))
                        )
                    chains_t = einops.rearrange(
                        torch.from_numpy(chains_trajs).float().to(self.device),
                        "s e t h d -> (s e) t h d",
                    )
                    chains_ts = torch.split(chains_t, self.logprob_batch_size, dim=0)
                    logprobs_trajs = np.empty(
                        (
                            0,
                            self.model.ft_denoising_steps,
                            self.horizon_steps,
                            self.action_dim,
                        )
                    )
                    for obs, chains in zip(obs_ts, chains_ts):
                        logprobs = self.model.get_logprobs(obs, chains).cpu().numpy()
                        logprobs_trajs = np.vstack(
                            (
                                logprobs_trajs,
                                logprobs.reshape(-1, *logprobs_trajs.shape[1:]),
                            )
                        )

                    # normalize reward with running variance if specified
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_transpose.T

                    # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
                    state = self.get_state(obs_venv)
                    
                    obs_venv_ts = {
                        "state": torch.from_numpy(state).float().to(self.device)
                    }
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextvalues = (
                                self.model.critic(obs_venv_ts)
                                .reshape(1, -1)
                                .cpu()
                                .numpy()
                            )
                        else:
                            nextvalues = values_trajs[t + 1]
                        nonterminal = 1.0 - terminated_trajs[t]
                        # delta = r + gamma*V(st+1) - V(st)
                        delta = (
                            reward_trajs[t] * self.reward_scale_const
                            + self.gamma * nextvalues * nonterminal
                            - values_trajs[t]
                        )
                        # A = delta_t + gamma*lamdba*delta_{t+1} + ...
                        advantages_trajs[t] = lastgaelam = (
                            delta
                            + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                        )
                    returns_trajs = advantages_trajs + values_trajs

                # k for environment step
                obs_k = {
                    "state": einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                }
                chains_k = einops.rearrange(
                    torch.tensor(chains_trajs, device=self.device).float(),
                    "s e t h d -> (s e) t h d",
                )
                returns_k = (
                    torch.tensor(returns_trajs, device=self.device).float().reshape(-1)
                )
                values_k = (
                    torch.tensor(values_trajs, device=self.device).float().reshape(-1)
                )
                advantages_k = (
                    torch.tensor(advantages_trajs, device=self.device)
                    .float()
                    .reshape(-1)
                )
                logprobs_k = torch.tensor(logprobs_trajs, device=self.device).float()

                # Update policy and critic
                total_steps = self.n_steps * self.n_envs * self.model.ft_denoising_steps
                clipfracs = []
                for update_epoch in range(self.update_epochs):
                    # for each epoch, go through all data in batches
                    flag_break = False
                    inds_k = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]  # b for batch
                        batch_inds_b, denoising_inds_b = torch.unravel_index(
                            inds_b,
                            (self.n_steps * self.n_envs, self.model.ft_denoising_steps),
                        )
                        obs_b = {"state": obs_k["state"][batch_inds_b]}
                        chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]
                        chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]
                        returns_b = returns_k[batch_inds_b]
                        values_b = values_k[batch_inds_b]
                        advantages_b = advantages_k[batch_inds_b]
                        logprobs_b = logprobs_k[batch_inds_b, denoising_inds_b]

                        # get loss
                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            eta,
                        ) = self.model.loss(
                            obs_b,
                            chains_prev_b,
                            chains_next_b,
                            denoising_inds_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                            reward_horizon=self.reward_horizon,
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )
                        clipfracs += [clipfrac]

                        # update policy and critic
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        if self.learn_eta:
                            self.eta_optimizer.zero_grad()
                        loss.backward()
                        if self.itr >= self.n_critic_warmup_itr:
                            if self.max_grad_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.actor_ft.parameters(), self.max_grad_norm
                                )
                            self.actor_optimizer.step()
                            if self.learn_eta and batch % self.eta_update_interval == 0:
                                self.eta_optimizer.step()
                        self.critic_optimizer.step()
                        log.info(
                            f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}"
                        )

                        # Stop gradient update if KL difference reaches target
                        if self.target_kl is not None and approx_kl > self.target_kl:
                            flag_break = True
                            break
                    if flag_break:
                        break

                # Explained variation of future rewards using value function
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

                # 每50步打印一次奖励统计信息
                #if step % 50 == 0:
                    #print(f"\nReward Statistics:")
                    #print(f"Reward trajs shape: {reward_trajs.shape}")
                    #print(f"Raw Reward trajs mean: {np.mean(reward_trajs)}")
                    #print(f"Raw Reward trajs max: {np.max(reward_trajs)}")
                    #print(f"Raw Reward trajs min: {np.min(reward_trajs)}")
                    #if self.reward_scale_running:
                    #print(f"Running reward scaler stats:")
                    #print(f"Mean: {self.running_reward_scaler.mean}")
                    #print(f"Var: {self.running_reward_scaler.var}")

                network_update_time = time.time() - update_start

            # Plot state trajectories (only in D3IL)
            if (
                self.itr % self.render_freq == 0
                and self.n_render > 0
                and self.traj_plotter is not None
            ):
                self.traj_plotter(
                    obs_full_trajs=obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )

            # Update lr, min_sampling_std
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                # 在保存模型时启用视频录制
                if hasattr(self.venv, 'envs'):
                    for env_ind in range(self.n_envs):
                        self.venv.envs[env_ind].env.record_video = True
                else:
                    self.venv.env.record_video = True
                self.save_model()
                # 保存完模型后关闭视频录制
                if hasattr(self.venv, 'envs'):
                    for env_ind in range(self.n_envs):
                        self.venv.envs[env_ind].env.record_video = False
                else:
                    self.venv.env.record_video = False

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["chains_trajs"] = chains_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=False,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | eta {eta:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {   "env_step_time": env_step_time,
                                "network_update_time": network_update_time,
                                "total_time": env_step_time + network_update_time,
                                "total env step": cnt_train_step,
                                "loss": loss,
                                "pg loss": pg_loss,
                                "value loss": v_loss,
                                "bc loss": bc_loss,
                                "eta": eta,
                                "approx kl": approx_kl,
                                "ratio": ratio,
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                                "diffusion - min sampling std": diffusion_min_sampling_std,
                                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                                "critic lr": self.critic_optimizer.param_groups[0][
                                    "lr"
                                ],
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
