"""LLM policy agent that directly acts based on recommendations from the LLM."""
import os
import time
import re
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from strategist.llm_text import LLMText
from strategist.openai_client import AzureOpenAIClient, OpenAIClient
from strategist.io_utils import read_yaml_file

# Import the describe_frame function from smartplay if available
from smartplay.crafter.crafter_env import describe_frame

# Define action names directly to avoid circular imports
ACTION_NAMES = ['noop', 'move_left', 'move_right', 'move_up', 'move_down', 'do', 'sleep', 'place_stone', 'place_table', 'place_furnace', 'place_plant', 'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe', 'make_wood_sword', 'make_stone_sword', 'make_iron_sword']

def load_prompt_from_file(file_path: str) -> Optional[str]:
    """Load prompt text from a file."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def reconstruct_info(data, view_size=(9, 7)):
    """Reconstruct info dictionary from data."""
    # Inventory-related keys
    if "inventory" not in data.keys():
        inventory_keys = [
            "inventory_health", "inventory_food", "inventory_drink", "inventory_energy",
            "inventory_sapling", "inventory_wood", "inventory_stone", "inventory_coal",
            "inventory_iron", "inventory_diamond", "inventory_wood_pickaxe", "inventory_stone_pickaxe",
            "inventory_iron_pickaxe", "inventory_wood_sword", "inventory_stone_sword", "inventory_iron_sword"
        ]
        inventory = {key.replace("inventory_", ""): data[key] for key in inventory_keys if key in data}
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
        achievements = {key.replace("achivement_", ""): data[key] for key in achievement_keys if key in data}
    else:
        achievements = data["achievements"]
    
    # Reconstruct the info dictionary
    info = {
        "inventory": inventory,
        "achievements": achievements,
        "discount": data.get("discount", 1.0),
        "semantic": data.get("semantic", None),
        "player_pos": data.get("player_pos", None),
        "player_facing": data.get("player_facing", None),
        "reward": data.get("reward", 0.0),
        "dead": data.get("done", False),
        "action": data.get("action", None),
        "view": view_size,
        "sleeping": False,  # Needed for describe_frame
    }
    return info

def describe_frame_from_data(data):
    """Describe a frame from raw data using the proper describe_frame function."""
    info = reconstruct_info(data)
    # Use the imported describe_frame function
    frame_description = describe_frame(info, action=None)
    return frame_description

class LLMPolicyAgent:
    def __init__(self, goal: str, difficulty: str = "easy", llm_model: str = "gpt-4o-mini", 
                 system_prompt: Optional[str] = None, use_azure: bool = True):
        """
        Initializes the LLM Policy Agent with a goal and an LLM model.

        Args:
            goal (str): The goal description.
            difficulty (str): The difficulty level ("easy" or "medium").
            llm_model (str): The ID of the LLM model to use.
            system_prompt (str, optional): System prompt for the LLM.
            use_azure (bool): Whether to use the Azure OpenAI client.
        """
        self.goal = goal
        self.difficulty = difficulty
        self.total_price = 0.0
        self.client = AzureOpenAIClient(llm_model) if use_azure else OpenAIClient(llm_model)
        
        # Set system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are an AI that controls an agent in the Crafter game environment. "
                "Your job is to decide the next action for the agent based on the current state description. "
                "Your response must contain only a single action name in the format 'ACTION:<action_name>'. "
                "For example: 'ACTION:move_left' or 'ACTION:do'. "
                "Select the action that best helps achieve the specified goal."
            )
        
        self.llm_model = LLMText(llm_model, system_prompt=system_prompt)
        self.action_names = ACTION_NAMES
        self.total_steps = 0
        self.successful_steps = 0
        
        # Load game context from the appropriate YAML file
        if difficulty == "easy":
            config_file = "prompts/crafter-mod-farm-easy.yaml"
        elif difficulty == "medium":
            config_file = "prompts/crafter-mod-farm-medium.yaml"
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        
        config = read_yaml_file(config_file)

        # Load the action recommendation prompt
        instruction_path = "prompts/reward_shaper/get_recommended_actions_prompt_1.txt"
        self.instructions = load_prompt_from_file(instruction_path)
        self.game_context = config.get("game_context", "")
        self.goal_context = config.get("goal_context", "")

    def _parse_action(self, response: str) -> str:
        """
        Parses the LLM response to extract the recommended action.

        Args:
            response (str): The response generated by the LLM.

        Returns:
            str: The recommended action.
        """
        # Case 1: Look for ACTION:<action> format
        pattern = r"ACTION:([a-zA-Z_]+)"
        match = re.search(pattern, response)
        if match:
            action = match.group(1).strip()
            if action in self.action_names:
                return action
        
        # Case 2: Just look for any exact action name in the text
        for action in self.action_names:
            if action in response:
                return action
        
        # Fallback: return noop if we can't parse a valid action
        print("WARNING: Could not parse a valid action from the LLM response. Defaulting to noop.")
        print(f"Response: {response}")
        return "noop"

    def get_action(self, state: Dict, verbose: bool = False) -> str:
        """
        Uses the LLM to recommend the next action for the current state.
        
        Args:
            state (Dict): The current state dictionary.
            verbose (bool): Whether to print verbose output.
            
        Returns:
            str: The recommended action name.
        """
        
        # Get state description using the proper describe_frame function
        state_description = describe_frame_from_data(state)
        
        # Construct prompt with game context and goal
        prompt = (
            f"# Goal\n{self.goal}\n\n"
            f"# Game Context\n{self.game_context}\n\n"
            f"# Current State\n{state_description}\n\n"
            f"# Instructions\n{self.instructions}"
        )
        
        # Generate response from the LLM
        response = self.llm_model.generate_response(
            text=prompt,
            max_new_tokens=200,
        )
        
        # Update price tracking
        self.total_price += self.llm_model.total_price
        
        if verbose:
            print("Prompt:\n", prompt)
            print("\nLLM Response:\n", response)
        
        # Parse the response to get the recommended action
        action = self._parse_action(response)
            
        self.total_steps += 1
        self.successful_steps += 1
            
        return action
        
    def convert_action_to_index(self, action_name: str) -> int:
        """
        Converts an action name to its corresponding index.
        
        Args:
            action_name (str): The name of the action.
            
        Returns:
            int: The index of the action.
        """
        try:
            return self.action_names.index(action_name)
        except ValueError:
            print(f"Unknown action '{action_name}', defaulting to noop")
            return 0  # Default to noop (index 0)
            
    def reset_stats(self):
        """Resets the agent's statistics."""
        self.total_steps = 0
        self.successful_steps = 0
        
    def get_success_rate(self):
        """Returns the rate of successful action recommendations."""
        if self.total_steps == 0:
            return 0.0
        return self.successful_steps / self.total_steps 