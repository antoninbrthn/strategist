import copy
import numpy as np
import torch as th

from typing import Optional, Callable, Tuple
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor

class CustomPPO(PPO):
    """
    A subclass of PPO that overrides the rollout collection
    to allow custom reward logic. It follows the original
    collect_rollouts() from stable_baselines3.common.on_policy_algorithm,
    with a small modification at the end to add custom rewards.
    """

    def __init__(
        self,
        custom_reward_func: Optional[Callable] = None,
        reward_mode: str = "direct",
        alpha: float = 1.0,
        alpha_schedule: Optional[Tuple[float, float, int, int]] = None,
        custom_reward_range: float = 1.0,
        decay_function: str = "linear",
        *args,
        **kwargs
    ):
        """
        :param custom_reward_func:
            A callable that takes a batch of observations (and possibly actions) and returns
            a custom scalar reward per step. Must output a shape compatible with
            `rollout_buffer.rewards`.
        :param reward_mode:
            One of ["direct", "diff", "only_custom", "only_natural"] indicating how to
            combine custom reward with the environment reward:
              - "direct": final_reward = env_reward + (custom_reward / alpha)
              - "diff": a difference-based scheme (see compute_custom_rewards)
              - "only_custom": ignore environment reward entirely
              - "only_natural": ignore custom reward entirely
        :param alpha:
            Scaling factor for custom rewards.
        :param alpha_schedule:
            A tuple (start, end, decay_start, decay_end) for linearly annealing alpha from start to end

        Other args/kwargs go directly to PPO's constructor.
        """
        super().__init__(*args, **kwargs)
        self.custom_reward_func = custom_reward_func
        self.reward_mode = reward_mode
        self.alpha = alpha
        self.alpha_schedule = alpha_schedule
        self.current_steps = 0
        self.custom_reward_range = custom_reward_range  # flat factor to size custom reward
        self.decay_function = decay_function  # shape of alpha decay


    def compute_custom_rewards(self, rollout_buffer: RolloutBuffer) -> None:
        """
        Compute and apply custom rewards based on the specified reward mode,
        modifying rollout_buffer.rewards in-place.
        """
        if self.custom_reward_func is None:
            return

        obs = copy.deepcopy(rollout_buffer.observations).squeeze()

        with th.no_grad():
            custom_rewards_tensor = self.custom_reward_func(obs)
        custom_rewards = custom_rewards_tensor.cpu(
        ).numpy().reshape(rollout_buffer.rewards.shape)

        if self.reward_mode == "direct":
            # Add custom reward to environment reward
            # rollout_buffer.rewards += custom_rewards / self.alpha
            rollout_buffer.rewards = self.custom_reward_range * self.alpha * custom_rewards + (1-self.alpha) * rollout_buffer.rewards

        elif self.reward_mode == "diff":
            # Example difference-based scheme:
            # r_diff[t] = gamma * r[t] - r[t-1]
            # Zero out across episode boundaries.
            # NOTE: If multiple envs / episodes in the buffer, more advanced logic might be needed.
            expanded = np.concatenate(
                [np.zeros((1,1)), custom_rewards[:-1]]  # leading zero
            )
            diff = rollout_buffer.gamma * custom_rewards - expanded
            diff *= (1 - rollout_buffer.episode_starts)  # zero if new episode
            # rollout_buffer.rewards += diff / self.alpha
            rollout_buffer.rewards = self.custom_reward_range * self.alpha * diff + (
                        1 - self.alpha) * rollout_buffer.rewards

        elif self.reward_mode == "only_custom":
            # Discard environment reward
            rollout_buffer.rewards[:] = custom_rewards

        elif self.reward_mode == "only_natural":
            # Do nothing (keep environment reward)
            pass

        else:
            raise ValueError(f"Unknown reward_mode={self.reward_mode}")


    def adjust_alpha(self):
        if self.alpha_schedule is not None:
            start_alpha, end_alpha, decay_start, decay_end = self.alpha_schedule
            if self.current_steps < decay_start:
                self.alpha = start_alpha
            elif self.current_steps < decay_end:
                if self.decay_function == "linear":
                    decay_func = lambda x: x
                elif "poly" in self.decay_function:  # "poly_X" -> x^X
                    decay_func = lambda x: x ** float(self.decay_function.split("_")[1])
                else:
                    raise ValueError(f"Unknown decay_function={self.decay_function}")
                self.alpha = start_alpha + (end_alpha - start_alpha) * (
                    decay_func((self.current_steps - decay_start) / (decay_end - decay_start))
                )
            else:
                self.alpha = end_alpha

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # Adjust alpha parameter
        self.adjust_alpha()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(
                        clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1
            self.current_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(
                            terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(
                new_obs, self.device))  # type: ignore[arg-type]

        self.compute_custom_rewards(rollout_buffer)


        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
