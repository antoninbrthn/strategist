"""Text-based reward shaper from the semantic view of the environment."""
import re
import os
import uuid
from typing import Any, Callable, List, Tuple, Optional, Union, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from smartplay.crafter.crafter_env import describe_frame
import pathlib
import pickle

from data.crafter import crafter
from strategist.config import EXPORT_DIR
from strategist.io_utils import read_yaml_file
from strategist.llm_text import LLMText
from strategist.rl_agent import RLAgent

data_yaml = read_yaml_file("data/SmartPlay/src/smartplay/crafter/crafter/data.yaml")
action_names = data_yaml["actions"]
MAX_SCORE = 5  # maximum score

def load_prompt_from_file(file_path: str) -> Optional[str]:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def reconstruct_info(data, view_size=(9, 7)):
    # Inventory-related keys
    if "inventory" not in data.keys():
        inventory_keys = [
            "inventory_health", "inventory_food", "inventory_drink", "inventory_energy",
            "inventory_sapling", "inventory_wood", "inventory_stone", "inventory_coal",
            "inventory_iron", "inventory_diamond", "inventory_wood_pickaxe", "inventory_stone_pickaxe",
            "inventory_iron_pickaxe", "inventory_wood_sword", "inventory_stone_sword", "inventory_iron_sword"
        ]
        inventory = {key.replace("inventory_", ""): data[key] for key in inventory_keys}
    else:
        inventory = data["inventory"]

    # Achievement-related keys
    if "achievements" not in data.keys():
        achievement_keys = [
            "achivement_collect_coal", "achivement_collect_diamond", "achivement_collect_drink",
            "achivement_collect_iron", "achivement_collect_sapling", "achivement_collect_stone",
            "achivement_collect_wood", "achivement_defeat_skeleton", "achivement_defeat_zombie",
            "achivement_eat_cow", "achivement_eat_plant", "achivement_make_iron_pickaxe",
            "achivement_make_iron_sword", "achivement_make_stone_pickaxe", "achivement_make_stone_sword",
            "achivement_make_wood_pickaxe", "achivement_make_wood_sword", "achivement_place_furnace",
            "achivement_place_plant", "achivement_place_stone", "achivement_place_table",
            "achivement_wake_up"
        ]
        achievements = {key.replace("achivement_", ""): data[key] for key in achievement_keys}
    else:
        achievements = data["achievements"]
    
    # Reconstruct the info dictionary
    info = {
        "inventory": inventory,
        "achievements": achievements,
        "discount": data["discount"],
        "semantic": data["semantic"],
        "player_pos": data["player_pos"],
        "player_facing": data.get("player_facing", None),
        "reward": data["reward"],
        "dead": data.get("done", False),  # Mapping `done` to `dead` - default to False if not present
        "action": data.get("action", None),  # not used in description
        "view": view_size,
    }
    return info

def describe_frame_from_data(data):
    info = reconstruct_info(data)
    # info["action"] = action_names[info["action"]]
    # action = info["action"]
    info["sleeping"] = False
    frame_description = describe_frame(info, action=None)  # ask LLM to provide the recommended action
    return frame_description

def make_prompt_from_states(states: List[dict]):
    prompt = ''
    for i in range(len(states)):
        prompt += f"\nSTATE {i+1}:\n"
        prompt += describe_frame_from_data(states[i]) + '\n'
    return prompt

def load_all_human_demonstrations_states():
    """Episode info provides information about the episode and the index of the state in the episode"""
    path = "<Path to Crafter human experts dataset>"  # available on crafter project page: https://danijar.com/project/crafter/
    all_states = []
    episode_info = []
    for fn in os.listdir(path):
        data = np.load(os.path.join(path, fn))
        data_loaded = {k: data[k] for k in data.keys()}
        states = [{k: data_loaded[k][i] for k in data_loaded.keys()} for i in range(data_loaded["image"].shape[0])]
        all_states += states
        episode_info += [(fn, i) for i in range(data_loaded["image"].shape[0])]
    print(f'Loaded {len(all_states)} total states from human demonstrations.')
    return all_states, episode_info

def load_states_from_human_demonstrations(n_samples, seed=123, skip_ids=None):
    all_states, episode_info = load_all_human_demonstrations_states()
    np.random.seed(seed)
    idx = list(range(len(all_states)))
    if skip_ids is not None:
        episode_ids = [f"{info[0]}_{info[1]}" for info in episode_info]
        idx = [i for i in idx if episode_ids[i] not in skip_ids]
        print("Skipping states in:", list(skip_ids)[:3], "...")
        assert (len(idx) >= n_samples), "Not enough states to sample from."
    sample_indices = np.random.choice(idx, n_samples, replace=False)
    selected_states = [all_states[i] for i in sample_indices]
    selected_states_info = [episode_info[i] for i in sample_indices]
    return selected_states, selected_states_info


def load_all_states_from_trained_agents(n_episodes):
    def collect_trajectories(agent, env, n_episodes):
        trajectories = []
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            episode_trajectory = []
            while not done:
                action, _ = agent.model.predict(obs)
                image, reward, done, info = env.step(action)
                info.update({
                    "image": image,
                    "action": action,
                    "reward": reward,
                    "done": done,
                })
                episode_trajectory.append(info)
            trajectories.append(episode_trajectory)
        return trajectories

    # Load PPO agents
    config = {"model": "PPO", "model_params": {"policy": "CnnPolicy", }, "train_ts": 3_000_000,
              "log_interval": 50_000, "checkpoint_interval": 100_000, "n_eval_episodes": 10}

    if ("__version__" in crafter.__dict__) and ("surv" in crafter.__version__):
        print("Using `surv` PPO agents")
        agent_500k_path = "<Path to PPO checkpoints>"
        agent_3M_path = "<Path to PPO checkpoints>"
    else:
        print("Using `farm` PPO agents")
        agent_500k_path = "<Path to PPO checkpoints>"
        agent_3M_path = "<Path to PPO checkpoints>"

    base_env = crafter.Env()
    agent_500k = RLAgent(env=base_env, config=config, use_wandb=False)
    agent_500k.model = PPO.load(
        agent_500k_path,
        env=base_env)
    agent_3M = RLAgent(env=base_env, config=config, use_wandb=False)
    agent_3M.model = PPO.load(
        agent_3M_path,
        env=base_env)
    agents_dict = {"agent_500k": agent_500k, "agent_3M": agent_3M}

    print(f'Collecting {n_episodes} trajectories from PPO agents: {agents_dict.keys()}')

    # Collect trajectories
    all_trajectories = {}
    for agent_name, agent in agents_dict.items():
        trajectories = collect_trajectories(agent, base_env, n_episodes=max(1, n_episodes//len(agents_dict)))
        all_trajectories[agent_name] = trajectories

    # Extract states and create ids
    all_states = []
    episode_info = []
    for agent_name, trajectories in all_trajectories.items():
        for trajectory_idx, trajectory in enumerate(trajectories):
            traj_hash = uuid.uuid1().hex[:8]
            for frame_idx, state in enumerate(trajectory):
                state_id = (f"{agent_name}-{traj_hash}", frame_idx)
                all_states.append(state)
                episode_info.append(state_id)
    return all_states, episode_info


def load_states_from_trained_agents_trajectories(n_samples, seed=123):
    print(f'Collecting {n_samples} frames from agents')
    np.random.seed(seed)
    n_episodes = max(1, n_samples // 20)  # on average 20 frames per episode
    all_states, episode_info = load_all_states_from_trained_agents(n_episodes=n_episodes)
    sample_indices = np.random.choice(len(all_states), n_samples, replace=False)
    selected_states = [all_states[i] for i in sample_indices]
    selected_states_info = [episode_info[i] for i in sample_indices]
    return selected_states, selected_states_info


def load_states_from_trained_agents_contrast(n_sapling_contrast):
    print(f'Extending the dataset with {n_sapling_contrast} sapling contrast frames')
    # Collect sapling contrast pairs
    sapling_contrast_states = []
    sapling_contrast_info = []
    sapling_changes = []
    while len(sapling_contrast_states) < n_sapling_contrast:
        print(f'Found {len(sapling_contrast_states)}/{n_sapling_contrast} sapling contrast pairs.')
        all_states, episode_info = load_all_states_from_trained_agents(n_episodes=5)
        for t in range(len(all_states) - 1):
            state_t = all_states[t]
            state_t1 = all_states[t + 1]
            sapling_t = state_t["inventory"].get("sapling", 0)
            sapling_t1 = state_t1["inventory"].get("sapling", 0)
            if (sapling_t != sapling_t1) & (not state_t['done']):
                sapling_change = f'{sapling_t}->{sapling_t1}'
                sapling_changes.append(sapling_change)
                sapling_contrast_states.append(state_t)
                sapling_contrast_states.append(state_t1)
                sapling_contrast_info.append(episode_info[t])
                sapling_contrast_info.append(episode_info[t + 1])
                if len(sapling_contrast_states) >= n_sapling_contrast:
                    break
            if len(sapling_contrast_states) >= n_sapling_contrast:
                break
    # print value counts of sapling chance
    print('Sapling contrast counts:')
    pd.Series(sapling_changes).value_counts()
    return sapling_contrast_states, sapling_contrast_info


def load_states_from_trained_agents(n_samples, n_sapling_contrast=2000, seed=123):
    """Loads "n_samples" states from trained agents trajectories, including at least "n_sapling_contrast" frames contrasting saplings"""
    selected_states = []
    selected_states_info = []
    n_samples_traj = max(0, n_samples - n_sapling_contrast)

    if n_samples_traj>0:
        selected_states_traj, selected_states_info_traj = load_states_from_trained_agents_trajectories(n_samples_traj, seed=seed)
        selected_states += selected_states_traj
        selected_states_info += selected_states_info_traj

    if n_sapling_contrast>0:
        sapling_contrast_states, sapling_contrast_info = load_states_from_trained_agents_contrast(n_sapling_contrast)
        selected_states += sapling_contrast_states
        selected_states_info += sapling_contrast_info

    return selected_states, selected_states_info


class TextRewardShaperScore:
    def __init__(self, goal: str, llm_model: str = "meta-llama/Llama-3.2-3B-Instruct",
                 transform: Optional[Callable[[Image.Image], torch.Tensor]] = None, system_prompt: Optional[str] = None,
                 reward_model_architecture: str = "resnet18", n_states_per_prompt=5,
                 normalization="sigmoid", n_votes=1, prompt_version_actions=1, prompt_version_scores=1,
                 n_samples=100, train_reward_epochs=30, n_sapling_contrast=0, mixup_factor=10,
                 export_root=None, random_seed=123, early_stopping=False,
                 human_annotation_ratio=0.5):
        """
        Initializes the RewardShaper with a goal and a VLM model.

        Args:
            goal (str): The goal description.
            llm_model (str): The ID of the VLM model to use.
            reward_model_architecture (str): The architecture of the reward model. Must in torchvision's models. Tested: resnet18, mobilenet_v3_small, mobilenet_v3_large, shufflenet_v2_x1_0
        """
        self.goal = goal
        self.llm_model = LLMText(llm_model, system_prompt=system_prompt)
        self.reward_model = None  
        self.action_names = ['noop', 'move_left', 'move_right', 'move_up', 'move_down', 'do', 'sleep', 'place_stone', 'place_table', 'place_furnace', 'place_plant', 'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe', 'make_wood_sword', 'make_stone_sword', 'make_iron_sword']
        # self._make_general_prompt()
        # Define image transformations
        self.transform = transform
        self.reward_model_architecture = reward_model_architecture
        self.trained = False  # Flag to indicate if the reward model has been trained (or loaded)
        self._norm_mode = normalization
        self.n_states_per_prompt = n_states_per_prompt
        self.n_votes = n_votes
        self.n_samples = n_samples  # number of frames to annotate
        self.n_sapling_contrast = n_sapling_contrast
        self.train_reward_epochs = train_reward_epochs  # number of epochs to train the reward model
        self.prompt_version_actions = prompt_version_actions
        self.prompt_version_scores = prompt_version_scores
        self.random_seed = random_seed
        self.early_stopping = early_stopping
        # self.use_human_demonstrations = use_human_demonstrations
        self.human_annotation_ratio = human_annotation_ratio  # ratio of human annotations to use vs from trained PPO agents
        self.mixup_factor = mixup_factor  # factor by which to extend the training data

        # Set annotation path
        self.export_root = export_root if export_root is not None else os.path.join(EXPORT_DIR, "annotations")
        self.goal_alias = re.sub(r'[>;(),]', '_', self.goal.lower().replace(' ', '-'))[:100]
        self.llm_model_alias = self.llm_model.model_id.split('/')[-1]
        self.export_dir = os.path.join(self.export_root, f"{self.llm_model_alias}_{self.goal_alias}")
        os.makedirs(self.export_dir, exist_ok=True)
        print(f"Export dir: {self.export_dir}")
        self.annotated_states = self.init_annotations()

    def _get_one_score(self, states: List[dict], verbose=False) -> List[int]:
        """
        Uses the VLM to compare assign scores to N states to the goal. Each state should be passed
        as dictionary containing at least the following attributes: `semantic`, `player_pos`.

        Args:
            states (List(dict)): List of states to compare.

        Returns:
            list of int: Scores assigned to each state.
        """
        # Generate prompts for analyzing each state
        # if reflect_prompt:
        prompt_file = f"prompts/reward_shaper/get_scores_prompt_{self.prompt_version_scores}.txt"
        prompt_text = load_prompt_from_file(prompt_file)
        prompt = prompt_text.format(goal=self.goal, n_states=len(states))

        # Get sem info
        state_prompt = make_prompt_from_states(states)
        combined_prompt = prompt + "\n---START---\n" + state_prompt

        response = self.llm_model.generate_response(
            text=combined_prompt,
            max_new_tokens=1000,
        )
        if verbose or (np.random.rand() < 0.02):
            print(combined_prompt)
            print(response)
        # Parse the response to get preference
        scores = self._parse_scores(response)
        return scores

    def get_config(self):
        keys = [
            "goal",
            "n_states_per_prompt",
            "n_votes",
            "n_samples",
            "train_reward_epochs",
            "random_seed",
            "early_stopping",
            "goal_alias",
            "llm_model_alias",
        ]
        return dict((key, self.__dict__[key]) for key in keys)

    def _parse_actions(self, response: str) -> List[str]:
        """
        Parses the LLM response to extract recommended actions.

        Args:
            response (str): The response generated by the LLM.

        Returns:
            List[str]: A list of recommended actions for each state.
        """
        # Define a regex pattern to extract the ACTIONS line
        pattern = r"ACTIONS:([a-zA-Z_,]+)"
        if not response:
            print("WARNING: LLM response is empty.")
            return None
        match = re.search(pattern, response.split("---START---")[-1])

        if match:
            actions_str = match.group(1)
            actions = actions_str.split(',')
            actions = [action.strip() for action in actions]
            return actions
        else:
            # Handle cases where the pattern is not found
            print("WARNING: Could not parse actions from the LLM response.")
            return None

    def init_annotations(self):
        # Load existing annotations if available (dictionary)
        annotation_file = os.path.join(self.export_dir, "annotations.pkl")
        if os.path.exists(annotation_file):
            with open(annotation_file, 'rb') as f:
                annotated_states = pickle.load(f)
            print(f"Loaded {len(annotated_states)} existing annotations from {annotation_file}")
        else:
            annotated_states = {}
        return annotated_states

    def summarize_annotations(self):
        scores = [v['score'] for v in self.annotated_states.values()]
        actions = [v['action'] for v in self.annotated_states.values()]
        print("Scores:\n", pd.Series(scores).value_counts())
        print("Actions:\n", pd.Series(actions).value_counts())
        return

    def annotate_states(self, verbose=False):
        # Check if current annotations are less than n_samples
        if len(self.annotated_states) >= self.n_samples:
            print(f"{len(self.annotated_states)} states have been annotated.")
            return True
        n_remaining = self.n_samples - len(self.annotated_states)
        print(f"Annotating {n_remaining} states.")

        # Load states to annotate
        n_remaining_human = int(n_remaining * self.human_annotation_ratio)
        n_remaining_trained = n_remaining - n_remaining_human
        print('Annotation split:', n_remaining_human, 'human demonstrations,', n_remaining_trained, 'trained agents')
        human_states, human_episode_info = load_states_from_human_demonstrations(n_remaining_human, seed=self.random_seed, skip_ids=self.annotated_states.keys())
        trained_states, trained_episode_info = load_states_from_trained_agents(n_remaining_trained, n_sapling_contrast=self.n_sapling_contrast, seed=self.random_seed)
        states = human_states + trained_states
        episode_info = human_episode_info + trained_episode_info
        # states, episode_info = load_states_from_trained_agents(n_remaining, seed=self.random_seed)

        # shuffle indices
        np.random.seed(self.random_seed)
        ids = np.random.choice(len(states), len(states), replace=False)

        # export config
        with open(os.path.join(self.export_dir, "config.yaml"), 'w') as f:
            yaml.dump(self.get_config(), f)

        # Process by batches of self.n_states_per_prompt
        for i in tqdm(range(0, len(ids), self.n_states_per_prompt)):
            batch_ids = ids[i: i + self.n_states_per_prompt]
            batch = [states[j] for j in batch_ids]
            batch_info = [episode_info[j] for j in batch_ids]

            # Use get_average_score to get a single set of averaged scores for this batch
            batch_scores = self.get_scores(batch, verbose=verbose)
            # If get_average_score returned None or an empty list, we default to -1
            if (not batch_scores) or (len(batch_scores) != len(batch)):
                batch_scores = [-1.0] * len(batch)  # TODO: change to 0, or discard altogether

            batch_actions = self.get_recommended_actions(batch, verbose=verbose)
            # If get_recommended_actions returned None or an empty list, we default to "noop"
            if (not batch_actions) or (len(batch_actions) != len(batch)):
                batch_actions = ["noop"] * len(batch)

            for j, (state, score) in enumerate(zip(batch_info, batch_scores)):
                state_id = f"{state[0]}_{state[1]}"
                state = batch[j].copy()
                state.update({"score": score, "llm_action": batch_actions[j]})
                self.annotated_states[state_id] = state

                # Save annotations
                with open(os.path.join(self.export_dir, "annotations.pkl"), 'wb') as f:
                    pickle.dump(self.annotated_states, f)
            if i % 10 == 0:
                print(f"Total price: {self.llm_model.total_price}")
        return True




    def get_scores(self, states: List[dict], verbose=False) -> List[float]:
        """
        Calls `_get_one_score` n_votes times for the provided list of states (size n_states),
        and returns the element-wise average of these scores as a list of floats.
        """
        all_scores = []
        for _ in range(self.n_votes):
            scores_tensor = self._get_one_score(states, verbose=verbose)
            # scores_tensor is typically a torch.Tensor of shape [n_states]
            if scores_tensor is not None:
                all_scores.append(scores_tensor)

        if len(all_scores) == 0:
            # No valid scores found; return zero for each state
            return [0.0] * len(states)

        # Stack into shape [n_votes, n_states] and average
        final_scores = torch.stack(all_scores, dim=0).float().mean(dim=0)
        return final_scores.tolist()

    def get_recommended_actions(self, states: List[dict], verbose=False) -> List[str]:
        """
        Uses the LLM to recommend the next action for each of N states to achieve the goal.
        Each state should be passed as a dictionary containing at least the following attributes: `semantic`, `player_pos`.

        Args:
            states (List(dict)): List of states to evaluate.

        Returns:
            List[str]: Recommended actions for each state.
        """
        prompt_file = f"prompts/reward_shaper/get_recommended_actions_prompt_{self.prompt_version_actions}.txt"
        prompt_text = load_prompt_from_file(prompt_file)
        prompt = prompt_text.format(goal=self.goal, n_states=len(states))

        # Get semantic information from states
        state_prompt = make_prompt_from_states(states)
        combined_prompt = prompt + "\n---START---\n" + state_prompt

        # Generate response from the LLM
        response = self.llm_model.generate_response(
            text=combined_prompt,
            max_new_tokens=1000,
        )
        
        if verbose:
            print("LLM Response:\n", response)
        
        # Parse the response to get recommended actions
        actions = self._parse_actions(response)
        return actions


    def train_reward_model(self,
                           batch_size: int = 16,
                           ):
        """
        Trains a reward model using the collected preferences.

        Args:
            preferences (List[Tuple[Image.Image, Image.Image, int]]): A list of tuples containing pairs of state images and their preference.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.

        Returns:
            Callable[[Image.Image], float]: A function that inputs a state and outputs a reward.
        """
        assert len(self.annotated_states) > 0, "No annotated states found. Please call annotate_states() first."
        print(f"Training reward model from {len(self.annotated_states)} frames.")

        # dataset = ScoreDataset(self.annotated_states, transform=self.transform, norm_mode=self._norm_mode)
        dataset = ScoreDatasetMixup(self.annotated_states, transform=self.transform, norm_mode=self._norm_mode, mixup_factor=self.mixup_factor)
        # print some dataset stats
        all_scores = [v for _, v in dataset]
        all_states = [s for s, v in dataset]
        print(f"Average target score: {np.mean(all_scores)} +/- {np.std(all_scores)}")
        print(f"Min score: {np.min(all_scores)}, Max score: {np.max(all_scores)}")

        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_test_split = 0.2
        train_size = int(len(dataset) * (1 - train_test_split))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Use a pre-trained CNN as feature extractor
        reward_model = self.reward_model_architecture
        feature_extractor = models.__dict__[reward_model](pretrained=True)

        if 'mobilenet' in reward_model:  # mobilenet_v3_small, mobilenet_v3_large
            num_features = feature_extractor.classifier[-1].in_features
            feature_extractor.classifier[-1] = nn.Identity()
        else:  # resnet18, shufflenet_v2_x1_0
            num_features = feature_extractor.fc.in_features
            feature_extractor.fc = nn.Identity()

        # Define the reward model
        self.reward_model = RewardModel(feature_extractor, num_features, norm_mode=self._norm_mode)
        # to cuda
        self.reward_model.cuda()

        # Define optimizer and loss function
        optimizer = optim.Adam(self.reward_model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        # Training loop
        self.reward_model.train()
        prev_loss = np.inf
        for epoch in range(self.train_reward_epochs):
            # Train the model
            self.reward_model.train()
            train_loss = 0
            for img, target in train_loader:
                optimizer.zero_grad()

                reward = self.reward_model(img).to('cpu')

                loss = criterion(reward, target)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss /= len(train_loader)

            # Evaluate on test set
            self.reward_model.eval()
            with torch.no_grad():
                test_loss = 0
                rewards = []
                for img, target in test_loader:
                    reward = self.reward_model(img).to('cpu')
                    loss = criterion(reward, target)
                    test_loss += loss.item()
                    rewards.append(reward)
                test_loss /= len(test_loader)
            print(f"Epoch {epoch+1}/{self.train_reward_epochs}, Train loss: {train_loss} Test loss: {test_loss}")
            print("Average predicted reward: ", torch.cat(rewards).mean().item())

            if self.early_stopping:
                # stop if loss is not decreasing
                if test_loss >= prev_loss:
                    print('Early stopping.')
                    break
            prev_loss = test_loss

        self.trained = True

        self.save_reward_model()
        return True

    def load_reward_model(self, model_path: str = None):
        """
        Loads a pre-trained reward model from a file.

        Args:
            model_path (str): The path to the reward model file.
        """
        if model_path is None:
            model_path = os.path.join(self.export_dir, "reward_model.pkl")
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}. Train a model first by calling train_reward_model().")
            return False
        self.reward_model = torch.load(model_path)
        self.trained = True
        print(f"Reward model loaded from {model_path}")
        return True

    def save_reward_model(self, model_path: str = None):
        """
        Saves the trained reward model to a file.

        :param model_path: The path to save the model.
        """
        if model_path is None:
            model_path = os.path.join(self.export_dir, "reward_model.pkl")

        torch.save(self.reward_model, model_path)

        # save traced model
        example_input = torch.randn(1, 3, 64, 64)
        traced_model = torch.jit.trace(self.reward_model, example_input)
        traced_model_export_path = os.path.join(os.path.dirname(model_path), "reward_model_general.pt")
        traced_model.save(traced_model_export_path)
        print(f"Reward model saved to {model_path}")
        print(f"Reward model saved in traced format to {traced_model_export_path}")
        return

    def get_reward_function(self) -> Callable[[Union[np.ndarray, torch.Tensor]], float]:
        # Return the reward model as a callable
        def reward_function(state: Union[np.ndarray, torch.Tensor]) -> Union[float, torch.Tensor]:
            self.reward_model.eval()
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    # Convert NumPy array to tensor
                    state_tensor = torch.from_numpy(state).float()
                elif isinstance(state, torch.Tensor):
                    # Ensure tensor is on the correct device and is of type float
                    state_tensor = state.float()
                else:
                    raise TypeError(f"Unsupported input type: {type(state)}")

                # Apply transformation if provided
                # ensure normalisation
                if state_tensor.max() > 1:
                    state_tensor /= 255
                if self.transform is not None:
                    img = self.transform(state_tensor)
                else:
                    img = state_tensor
                # unsqueeze to add batch dimension if not present
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)

                # Make sure the model is on the correct device (cuda if available)
                if torch.cuda.is_available():
                    self.reward_model.cuda()
                    img = img.cuda()
                # Compute reward using the reward model
                reward = self.reward_model(img)
                return reward.squeeze()

        return reward_function



    def _parse_scores(self, response: str) -> torch.Tensor:
        """
        Parses the VLM's response to extract the scores. Format of the scores in the response:
        SCORES:<score for state 1>,<score for state 2>,...,<score for state N>

        Args:
            response (str): The response from the VLM.

        Returns:
            torch.Tensor of length N: The scores assigned to each state.
        """
        match = re.search(r"SCORES:\s*([0-5](?:,\s*[0-5])*)", response.split("---START---")[-1])

        # ignore capitalization
        if match is None:
            print("WARNING: Could not find scores in the last line of the response.")
            print(response.split("---START---")[-1])
            return None
        scores = match.group(0).split(":")[1].split(",")
        scores = [int(score) for score in scores if score != ""]
        return torch.tensor(scores)


    def build_scores(self, observations, num_samples: int = 10, verbose=False):
        """
        Builds a scores dataset from a list of states.

        Args:
            observations (List[dict]): List of state dictionaries.
            num_samples (int): Number of samples to draw from observations.

        Returns:
            torch.Tensor: A concatenated tensor of average scores (floats) for each sampled state.
        """
        scores = []
        # Randomly pick state indices
        ids = np.random.choice(len(observations), size=num_samples, replace=False)

        # Process by batches of self.n_states_per_prompt
        for i in range(0, num_samples, self.n_states_per_prompt):
            batch = [observations[j] for j in ids[i : i + self.n_states_per_prompt]]

            # Use get_average_score to get a single set of averaged scores for this batch
            batch_scores_list = self.get_scores(batch, verbose=verbose)
            # If get_average_score returned None or an empty list, we default to -1
            if not batch_scores_list:
                batch_scores_tensor = torch.tensor([-1.0] * len(batch), dtype=torch.float)
            else:
                batch_scores_tensor = torch.tensor(batch_scores_list, dtype=torch.float)

            scores.append(batch_scores_tensor)

        # Concatenate all batches into a single tensor
        return torch.cat(scores, dim=0)
    
    # def _make_general_prompt(self):
    #     root = pathlib.Path(__file__).parent
    #     root = root.parent / "data" / "SmartPlay" / "src" / "smartplay" / "crafter"
    #     with open(root / "assets/crafter_ctxt.pkl", 'rb') as f: # Context extracted using text-davinci-003 following https://arxiv.org/abs/2305.15486
    #         CTXT = pickle.load(f)
    #     CTXT = CTXT.replace("DO NOT answer in LaTeX.", "")
    #     CTXT = CTXT.replace("Move Up: Flat ground above the agent.", "Move North: Flat ground north of the agent.")
    #     CTXT = CTXT.replace("Move Down: Flat ground below the agent.", "Move South: Flat ground south of the agent.")
    #     CTXT = CTXT.replace("Move Left: Flat ground left to the agent.", "Move West: Flat ground west of the agent.")
    #     CTXT = CTXT.replace("Move Right: Flat ground right to the agent.", "Move East: Flat ground east of the agent.")
    #     self.desc = CTXT

class PreferenceDataset(Dataset):
    def __init__(self, preferences: List[Tuple[dict, dict, int]], transform=None):
        """
        Dataset for preference pairs.

        Args:
            preferences (List[Tuple[Image.Image, Image.Image, int]]): List of preference pairs and labels.
            transform: Transformations to apply to images.
        """
        self.preferences = preferences
        self.transform = transform

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        s1, s2, target = self.preferences[idx]
        img1, img2 = Image.fromarray(s1["image"]), Image.fromarray(s2["image"])
        # convert to tensor
        img1 = transforms.ToTensor()(img1)
        img2 = transforms.ToTensor()(img2)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        target = torch.tensor([target], dtype=torch.float)
        return img1, img2, target

class ScoreDataset(Dataset):
    def __init__(self, annotations: dict, transform=None, norm_mode="sigmoid"):
        """
        Dataset for preference pairs.

        Args:
            preferences (List[Tuple[Image.Image, Image.Image, int]]): List of preference pairs and labels.
            transform: Transformations to apply to images.
        """
        self.annotations_list = [dict(v, **{"id": k}) for k, v in annotations.items()]
        self.transform = transform
        self.norm_mode = norm_mode

    def __len__(self):
        return len(self.annotations_list)

    def __getitem__(self, idx):
        state = self.annotations_list[idx]
        # Normalise score to [-1, 1]
        img, target = Image.fromarray(state["image"]), self.normalise_score(state["score"])
        # convert to tensor
        img = transforms.ToTensor()(img)
        if self.transform:
            img = self.transform(img)
        target = torch.tensor([target], dtype=torch.float)
        return img, target

    def normalise_score(self, score):
        if self.norm_mode == "sigmoid":
            return np.clip(score / MAX_SCORE, 0, 1)
        elif self.norm_mode == "tanh":
            # Clipping to account for frames annotates with "-1"
            return np.clip((score - MAX_SCORE/2) / (MAX_SCORE/2), -1, 1)
        else:
            raise ValueError(f"Unsupported normalization mode: {self.norm_mode}")

class ScoreDatasetMixup(Dataset):
    def __init__(self, annotations: dict, transform=None, norm_mode="sigmoid", mixup_factor=1):
        """
        Dataset for preference pairs.
        Perform data augmentation by adding mixed-up images and labels.

        Args:
            preferences (List[Tuple[Image.Image, Image.Image, int]]): List of preference pairs and labels.
            transform: Transformations to apply to images.
        """
        self.annotations_list = [dict(v, **{"id": k}) for k, v in annotations.items()]
        self.transform = transform
        self.norm_mode = norm_mode
        self.mixup_factor = mixup_factor
        self.mixup_annotations = self.make_mixup_annotations()
        print(f'Made {len(self.mixup_annotations)} mixup annotations')

    def __len__(self):
        return len(self.annotations_list) + len(self.annotations_list) * self.mixup_factor

    def __getitem__(self, idx):
        if idx < len(self.annotations_list):
            state = self.annotations_list[idx]
            img, target = Image.fromarray(state["image"]), self.normalise_score(state["score"])
            img = transforms.ToTensor()(img)
        else:
            img, target = self.mixup_annotations[idx - len(self.annotations_list)]
        # Normalise score to [-1, 1]
        # convert to tensor
        if self.transform:
            img = self.transform(img)
        target = torch.tensor([target], dtype=torch.float)
        return img, target

    def make_mixup_annotations(self):
        mixup_annotations = []
        for _ in range((len(self.annotations_list) * self.mixup_factor)):
            # Randomly select two states
            idx1, idx2 = np.random.choice(len(self.annotations_list), 2, replace=False)
            state1, state2 = self.annotations_list[idx1], self.annotations_list[idx2]
            # Mix-up images
            img1, img2 = Image.fromarray(state1["image"]), Image.fromarray(state2["image"])
            img1 = transforms.ToTensor()(img1)
            img2 = transforms.ToTensor()(img2)
            img = img1 * 0.5 + img2 * 0.5
            # Mix-up scores
            target1 = self.normalise_score(state1["score"])
            target2 = self.normalise_score(state2["score"])
            target = target1 * 0.5 + target2 * 0.5
            mixup_annotations.append((img, target))
        return mixup_annotations

    def normalise_score(self, score):
        if self.norm_mode == "sigmoid":
            return np.clip(score / MAX_SCORE, 0, 1)
        elif self.norm_mode == "tanh":
            # Clipping to account for frames annotates with "-1"
            return np.clip((score - MAX_SCORE/2) / (MAX_SCORE/2), -1, 1)
        else:
            raise ValueError(f"Unsupported normalization mode: {self.norm_mode}")

class RewardModel(nn.Module):
    def __init__(self, feature_extractor: nn.Module, num_features: int, norm_mode: str = "sigmoid"):
        """
        Neural network for the reward model.

        Args:
            feature_extractor (nn.Module): Pre-trained feature extractor.
            num_features (int): Number of features from the extractor.
        """
        super(RewardModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(num_features, 1)
        self.norm_mode = norm_mode


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Reward value.
        """
        # make sure input is on the same device as the model
        x = x.to(next(self.parameters()).device)
        features = self.feature_extractor(x)
        reward = self.fc(features)
        if self.norm_mode == "sigmoid":
            reward = torch.sigmoid(reward)
        elif self.norm_mode == "tanh":
            reward = torch.tanh(reward)
        else:
            raise ValueError(f"Unsupported normalization mode: {self.norm_mode}")
        return reward