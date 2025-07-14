"""
Utilities to train RL agents with custom reward functions.
"""

import os

import PIL
import numpy as np
import torch
from stable_baselines3 import PPO, DQN
import random

import wandb
from stable_baselines3.common.callbacks import BaseCallback
import time

from strategist.utils import build_info_panel
from strategist.config import EXPORT_DIR, CONFIGS_DIR
from strategist.io_utils import read_yaml_file
from strategist.custom_ppo import CustomPPO
from strategist.custom_dqn import CustomDQN

def render_episode_gif(trajectory, episode_reward=None, path=os.path.join(EXPORT_DIR, "gifs"), fn=None):
    os.makedirs(path, exist_ok=True)
    states = []
    for prev_obs, action, obs, reward, done, info in trajectory:
        obs_upscale = obs.repeat(4, axis=0).repeat(4, axis=1)
        states.append(obs_upscale)
    import imageio
    if not fn:
        fn = "crafter.gif" if (episode_reward is None) else f"crafter_{episode_reward}.gif"

def render_episode_gif_annot(trajectory, llm_rewards, episode_reward=None, path=os.path.join(EXPORT_DIR, "gifs"), fn=None):

    frames_pil = []
    for step_idx, (prev_obs, action, obs, reward, done, info) in enumerate(trajectory):
        obs_upscale = obs.repeat(2, axis=0).repeat(2, axis=1)
        frame_arr = obs_upscale.copy()

        pil_frame = PIL.Image.fromarray(frame_arr.squeeze().astype(np.uint8))

        W, H = pil_frame.size
        panel_width = int(W)
        new_im = PIL.Image.new('RGB', (W + panel_width, H), color=(0,0,0))
        new_im.paste(pil_frame, (0,0))

        llm_reward = llm_rewards[step_idx]
        llm_reward_delta = llm_reward - llm_rewards[step_idx-1] if step_idx > 0 else 0
        msg_lines = [
            f"Step {step_idx}",
            f"Action = {action}",
            f"Env Reward = {reward:.2f}",
            f"Custom Reward  = {llm_reward:.2f}",
            f"d(Custom Reward) = {llm_reward_delta:.2f}",
        ]
        msg = "\n".join(msg_lines)

        info_panel = build_info_panel(panel_width, H, msg)
        new_im.paste(info_panel, (W, 0))

        frames_pil.append(new_im)

    os.makedirs(path, exist_ok=True)
    if not fn:
        fn = "crafter.gif" if (episode_reward is None) else f"crafter_{episode_reward}.gif"
    gif_path = os.path.join(path, fn)
    frames_pil[0].save(
        gif_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=300,
        loop=0,
    )
    print(f"Saved trajectory GIF: {gif_path}")


class WandbCallback(BaseCallback):
    def __init__(self,
                 eval_env,
                 n_eval_episodes=5,
                 log_interval=50_000,
                 checkpoint_interval=100_000,
                 save_path=EXPORT_DIR,
                 verbose=1):
        super(WandbCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.save_path = save_path
        self.save_counter = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            # Evaluate the model
            rewards = []
            llm_rewards = []
            episode_lengths = []
            saplings_collected = []
            sapling_to_cow = []
            gif_path = None
            for i in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                total_reward = 0
                trajectory = []
                all_achievements = []
                prev_sapling_count = 0
                total_saplings = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    next_obs, reward, done, info = self.eval_env.step(action)
                    trajectory.append((obs, action, next_obs, reward, done, {}))
                    obs = next_obs
                    sapling_count = info['inventory']['sapling']
                    if sapling_count > prev_sapling_count:
                        total_saplings += 1
                    prev_sapling_count = sapling_count
                    total_reward += reward
                rewards.append(total_reward)
                # sapling_to_cow.append(info['achievements']['sapling_to_cow'])
                sapling_to_cow.append(info['achievements'].get('sapling_to_cow', 0))
                episode_lengths.append(len(trajectory))
                all_achievements.append(info['achievements'])
                traj_obs = np.array([obs for obs, _, _, _, _, _ in trajectory])
                traj_obs_tensor = torch.as_tensor(traj_obs).float().permute(0, 3, 1, 2).to('cuda')
                if hasattr(self.model, 'custom_reward_func'):
                    traj_llm_rewards = self.model.custom_reward_func(traj_obs_tensor).cpu().detach().numpy()
                else:
                    traj_llm_rewards = np.zeros(len(trajectory)) - 1
                llm_rewards.append(np.mean(traj_llm_rewards))
                saplings_collected.append(total_saplings)

                # Save GIF for every other evaluation run
                if self.n_calls % 100_000 == 0:
                    if i%10== 0:
                        gif_path = os.path.join(self.save_path, "gifs")
                        fn = f"trajectory_{self.n_calls}_{i}.gif"
                        # render_episode_gif(trajectory, episode_reward=total_reward, path=gif_path, fn=fn)
                        render_episode_gif_annot(trajectory, llm_rewards=traj_llm_rewards, episode_reward=total_reward, path=gif_path, fn=fn)

            avg_reward = np.mean(rewards)
            avg_custom_reward = np.mean(llm_rewards)
            std_reward = np.std(rewards)
            std_custom_reward = np.std(llm_rewards)
            avg_length = np.mean(episode_lengths)
            std_length = np.std(episode_lengths)
            avg_achievements = {}
            for k in all_achievements[0]:
                values = [d[k] for d in all_achievements]
                avg_achievements[f"achievements/{k}_avg"] = np.mean(values)
            alpha = self.model.alpha if hasattr(self.model, 'alpha') else None
            wandb.log({"average_reward": avg_reward,
                        "avg_custom_reward": avg_custom_reward,
                        "std_reward": std_reward,
                        "std_custom_reward": std_custom_reward,
                        "average_episode_length": avg_length,
                        "avg_sapling_collected": np.mean(saplings_collected),
                        "std_sapling_collected": np.std(saplings_collected),
                        "avg_sapling_to_cow": np.mean(sapling_to_cow),
                        "std_sapling_to_cow": np.std(sapling_to_cow),
                        "std_episode_length": std_length,
                        "alpha": alpha,
                        "step": self.n_calls,
                        **avg_achievements
                        })

        if self.n_calls % self.checkpoint_interval == 0:
            # Save model checkpoint
            checkpoint_path = os.path.join(self.save_path, "checkpoints", f"checkpoint_{self.n_calls}.zip")
            os.makedirs(self.save_path, exist_ok=True)
            self.model.save(checkpoint_path, exclude=['custom_reward_func'])
            self.save_counter += 1
            if self.verbose > 0:
                print(f"Checkpoint saved to {checkpoint_path}")

        return True


def set_global_seed(seed: int):
    """Sets seeds for python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full reproducibility (potentially slow):
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

class RLAgent:
    """
    Language-conditioned RL agent.
    args:
    - env: the environment
    - config: configuration of the RL algorithm to solve the env
    - reward_func: function that takes a state and returns a reward
    - demonstrations: function that takes a state and returns an action (not supported yet)
    - export_path: path to export the trained model
    """

    def __init__(self, env, config, demonstrations=None, export_path=EXPORT_DIR, use_wandb=True, run_id=None, custom_reward_func=None):
        self.env = env
        self.config = config
        # Retrieve a seed from the config (or default to something)
        self.seed = config.get("seed", 0)
        set_global_seed(self.seed)
        self.model_name = config.get("model", "PPO")
        self.model_params = config.get("model_params", {})
        if "Custom" in self.model_name:
            self.model_params["custom_reward_func"] = custom_reward_func
        self.train_ts = config.get("train_ts", 100_000)
        self.log_interval = config.get("log_interval", 10_000)
        self.checkpoint_interval = config.get("checkpoint_interval", 100_000)
        self.n_eval_episodes = config.get("n_eval_episodes", 5)

        self.demonstrations = demonstrations  # Not implemented
        self.model = eval(f"{self.model_name}")(env=self.env, **self.model_params)
        self.use_wandb = use_wandb
        self.alias = f"crafter_{self.model_name}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.export_path = export_path

        if self.use_wandb:
            print("Logging in to wandb")
            # check that
            name = self.alias if run_id is None else None
            tags = self.config.get('wandb_tags', [])
            wandb.init(project="strategist", config=self.config, id=run_id, name=name, resume="allow", tags=tags)

    def train(self):
        eval_env = self.env  # Assuming same env for evaluation; modify as needed
        callback = WandbCallback(eval_env, n_eval_episodes=self.n_eval_episodes,
                                 log_interval=self.log_interval, checkpoint_interval=self.checkpoint_interval,
                                 save_path=os.path.join(self.export_path, self.alias))
        self.model.learn(total_timesteps=self.train_ts, callback=callback, progress_bar=True)
        wandb.finish()  # stop wandb logging
        print('Training complete. Stopped wandb logging.')

    def run_one_episode(self):
        prev_obs = self.env.reset()
        trajectory = []
        episode_reward = 0
        done = False
        while not done:
            action, _states = self.model.predict(prev_obs)
            obs, reward, done, info = self.env.step(action)
            trajectory.append((prev_obs, action, obs, reward, done, info))
            prev_obs = obs
            episode_reward += reward
        print(f"Episode reward: {episode_reward}")
        return trajectory, episode_reward, info

    def save_model(self):
        os.makedirs(os.path.join(EXPORT_DIR, "rl-models"), exist_ok=True)
        fn = f"{self.model_name}_{self.train_ts}"
        self.model.save(os.path.join(EXPORT_DIR, "rl-models", fn))

    def load_from_checkpoint(self, ckpt_name):
        ckpt_path = os.path.join(EXPORT_DIR, "rl-models", ckpt_name)
        self.model = eval(f"{self.model_name}").load(ckpt_path, env=self.env)


def example_usage():
    import crafter
    # Example usage:
    config_file = 'rl_agent.yaml'
    config_path = os.path.join(CONFIGS_DIR, config_file)
    config = read_yaml_file(config_path)
    # custom_func_test = lambda x: x.mean() - 50
    custom_func_test = None
    env = crafter.Env(custom_reward_func=custom_func_test)
    agent = RLAgent(env, config)
    # load from checkpoint
    # agent.load_from_checkpoint(ckpt_name="crafter_ppo_cnn_2100000")
    agent.train()
    trajectory, episode_reward = agent.run_one_episode()
    render_episode_gif(trajectory, episode_reward)
    # save model  # agent.save_model()

