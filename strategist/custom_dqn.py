import torch
from stable_baselines3 import DQN


class CustomDQN(DQN):
    """
    Custom DQN implementation that modifies rewards using a custom reward function
    before invoking the base class training method.
    """

    def __init__(self, custom_reward_func, alpha=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_reward_func = custom_reward_func
        self.alpha = alpha  # weighting parameter for the real reward

    def train(self, gradient_steps: int, batch_size: int) -> None:
        # Modify rewards in the replay buffer before training
        self.modify_rewards_in_replay_buffer(batch_size)

        # Proceed with the standard DQN training process
        super().train(gradient_steps, batch_size)

    def modify_rewards_in_replay_buffer(self, batch_size: int):
        # Sample a batch of transitions
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

        with torch.no_grad():
            # Compute custom rewards based on observations
            obs = replay_data.observations.clone().to('cuda')
            custom_rewards = self.compute_custom_rewards(obs)
            custom_rewards = custom_rewards.to(self.device).reshape(replay_data.rewards.shape)
            # Combine custom rewards with original rewards
            modified_rewards = custom_rewards / self.alpha + replay_data.rewards

        # Update the rewards in the replay buffer
        for i in range(batch_size):
            self.replay_buffer.add(replay_data.observations[i].cpu().numpy(),
                                   replay_data.next_observations[i].cpu().numpy(),
                                   replay_data.actions[i].cpu().numpy(),
                                   modified_rewards[i].cpu().numpy(),
                                   replay_data.dones[i].cpu().numpy(),
                                   [{}],  # Optional info dictionary
                                   )

    def compute_custom_rewards(self, observations):
        # Your custom reward logic goes here
        return self.custom_reward_func(observations)
