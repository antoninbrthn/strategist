import json
import os
import pickle

import numpy as np

import crafter

from strategist.test_reward_function import test_reward_function
from strategist.config import Config
from strategist.conversation import Conversation
from strategist.io_utils import create_prompt_from_list
from strategist.openai_client import AzureOpenAIClient, OpenAIClient
from strategist.reward_shaper_text_score import TextRewardShaperScore
from strategist.rl_agent import RLAgent
from strategist.tree import Tree


class Strategist:
    def __init__(self, max_rounds=2, config_file=None, use_azure=False, sample_states=None):
        self.config = Config(config_file=config_file)
        self.client = (
            AzureOpenAIClient(self.config.model)
            if use_azure
            else OpenAIClient(self.config.model)
        )
        self.tree = Tree(self.config.tree_file)
        self.conversation = Conversation(self.config.convo_file)
        self.prompts = self.config.read_combined_prompts()
        self.total_price = 0
        self.max_rounds = max_rounds

        # For reward shaping and train an RL agent on the environment
        self.env_name = self.config.env_name
        self.reward_shaper = None
        self.sample_states = sample_states
        self.agent_config = None

    @property
    def output_dir(self):
        return os.path.join("runs", self.config.run_id)

    def initialize_tree(self):
        content_prompt = create_prompt_from_list(
            [self.prompts["strategy_context"], self.prompts["tree_commands"]]
        )
        prompt_list = []
        for s in ['goal_context', 'game_context']:
            if s in self.prompts:
                prompt_list.append(self.prompts[s])
        goal_prompt = create_prompt_from_list(prompt_list)

        messages = [
            {"role": "user", "content": content_prompt},
            {"role": "user", "content": goal_prompt},
        ]
        self.conversation.messages = messages
        print("--- Initial Prompt ----\n", messages)
        completion = self.client.get_completion(messages)
        self.update_price(completion)
        llm_response = completion.choices[0].message.content
        print("--- LLM Response ---\n", llm_response)
        self.tree.init_tree(overall_goal=goal_prompt)
        self.tree.apply_changes(llm_response)

        args = {
            "system_fingerprint": completion.system_fingerprint,
            "model": completion.model,
        }
        self.conversation.add_message("assistant", llm_response, **args)

    def run(self):
        if not self.tree.tree:
            self.initialize_tree()

        for i in range(self.max_rounds):
            print(f"ROUND {i}")
            # self.tree.compute_tree_metrics()
            best_leaf_id = self.tree.find_most_promising_leaf()
            root_to_leaf_path = self.tree.extract_root_to_leaf_path(best_leaf_id)

            path_str = "\n".join(
                [
                    f"Node {node['id']}: Goal: {node['goal']}, Parent node: {node['parent']} (Feasibility: {node['feasibility']}, Value: {node['value']})"
                    for node in root_to_leaf_path
                ]
            )

            new_user_prompt = create_prompt_from_list(
                [
                    # self.prompts["goal_context"],  # goal context already in the root_to_leaf_path
                    self.prompts["provide_best_node"].format(
                        root_to_leaf_path=path_str
                    ),
                ]
            )
            self.conversation.add_message("user", new_user_prompt)

            print("--- User Prompt ----\n", new_user_prompt)
            completion = self.client.get_completion(self.conversation.messages)
            self.update_price(completion)
            llm_response = completion.choices[0].message.content
            args = {
                "system_fingerprint": completion.system_fingerprint,
                "model": completion.model,
            }
            print("--- LLM Response ---\n", llm_response, args)
            self.conversation.add_message("assistant", llm_response, **args)

            # Check for the END command in the LLM response
            if "END_TREE" in llm_response:
                print("Ending iterations as per LLM response.")
                break

            if self.tree.is_read_tree_command(llm_response):
                tree_str = self.tree.get_tree_string()
                self.conversation.add_message("user", tree_str)
            self.tree.apply_changes(llm_response)

    def update_price(self, completion):
        price = self.client.get_price(completion)
        self.total_price += price
        print(f"Request price: {price}")
        print(f"Total price: {self.total_price}")

    def load_run(self, run_id):
        self.config.run_id = run_id
        self.tree = Tree(self.config.tree_file)
        self.conversation = Conversation(self.config.convo_file)

    def export_optimal_path(self, run_dir):
        # Find the most promising leaf
        best_leaf_id = self.tree.find_most_promising_leaf()
        # Extract the path from root to the best leaf
        root_to_leaf_path = self.tree.extract_root_to_leaf_path(best_leaf_id)

        # Convert the path to a JSON serializable format
        path_data = [
            {
                "id": node["id"],
                "goal": node["goal"],
                "parent": node["parent"],
                "feasibility": node["feasibility"],
                "value": node["value"],
            }
            for node in root_to_leaf_path
        ]

        # Define the output directory and file path
        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(self.output_dir, "optimal_path.json")

        # Write the path data to a JSON file
        with open(output_file, "w") as f:
            json.dump(path_data, f, indent=4)

        # Print the path data
        print("--- Optimal Path ---")
        print(json.dumps(path_data, indent=4))

    def do_node_v2(self, node_id=None, use_wandb=True, verbose=False):
        """Trains an RL agent on a particular node."""
        # Get the node to train on
        if node_id is None:  # default to the most promising leaf
            node_id = self.tree.find_most_promising_leaf()
        print(f"Running on node ID: {node_id}")
        root_to_leaf_path = self.tree.extract_root_to_leaf_path(node_id)
        leaf_node = root_to_leaf_path[-1]
        # goal = ">".join([h['goal'] for h in root_to_leaf_path[1:]])
        goal = get_goal_label(nodes=root_to_leaf_path[1:])
        print("Goal:", goal)
        # goal = leaf_node["goal"]

        node_folder = f'node{node_id}'
        node_path = os.path.join(self.output_dir, node_folder)
        os.makedirs(node_path, exist_ok=True)
        # dump goal.txt
        with open(os.path.join(node_path, "goal.txt"), "w") as f:
            f.write(goal)

        # Add node to self.agent_config
        self.agent_config.update({"node_id": node_id})
        self.agent_config.update({"goal": goal})
        print(self.agent_config)

        agent = train_agent(goal=goal, agent_config=self.agent_config, env_name=self.env_name, use_wandb=use_wandb, verbose=verbose)

        return agent

def get_env(env_name, reward_func=None, difficulty='easy', seed=None):
    """Initialize the environment based on the environment name."""
    if (env_name == 'crafter') and (difficulty == 'surv'):
        assert ("__version__" in crafter.__dict__) and ("surv" in crafter.__version__), "Please install the survival mod of Crafter (branch `survival`)."
        print('Creating CraterSurv env with seed =', seed)
        env = crafter.Env(custom_reward_func=reward_func, seed=seed)
    elif (env_name == 'crafter') and (difficulty == 'og'):
        assert ("__version__" in crafter.__dict__) and ("og" in crafter.__version__), "Please install the og version of Crafter (branch `og`)."
        print('Creating CraterOG env with seed =', seed)
        env = crafter.Env(custom_reward_func=reward_func, seed=seed)
    elif env_name == 'crafter':
        print('Creating env with seed =', seed)
        env = crafter.Env(custom_reward_func=reward_func, difficulty=difficulty, seed=seed)
    else:
        raise ValueError(f"Environment {env_name} not supported.")
    return env


def get_goal_label(nodes):
    """Build the goal title from a series of nodes.
    For each node extract the 'goal'.
    From node1, node2, etc, the goal should be: <goal1> by following the strategy of <goal2> by following the strategy of <goal3> etc.
    If a node is of the form "Plan(subgoal1, subgoal2, ...)", then it should be turned into: "first subgoal1, then subgoal2, ..."
    """
    goal_parts = []
    for node in nodes:
        if node['goal'].startswith("Plan("):
            subgoals = node['goal'][5:-1].split(";")
            subgoal_str = "first " + ", then ".join(subgoals)
            goal_parts.append(subgoal_str)
        else:
            goal_parts.append(node['goal'])
    goal = " by following the strategy of ".join(goal_parts)
    return goal


def train_agent(goal, agent_config, env_name, use_wandb=True, verbose=False):
    if "Custom" in agent_config["model"]:
        # Reward shaping with goal
        reward_config = agent_config.get("reward_config", {})
        reward_shaper = TextRewardShaperScore(goal, **reward_config)
        if not reward_shaper.load_reward_model():
            reward_shaper.annotate_states(verbose=verbose)
            reward_shaper.summarize_annotations()  # show the distribution of scores
            reward_shaper.train_reward_model()
        reward_func = reward_shaper.get_reward_function()

        # Test reward function on some PPO agents
        test_path = os.path.join(reward_shaper.export_dir, "test_reward_model")
        if (not os.path.exists(test_path)) or (len(os.listdir(test_path)) == 0):
            print('Testing reward function')
            os.makedirs(test_path, exist_ok=True)
            test_reward_function(reward_model=reward_func, output_path=test_path)
        else:
            print("Test reward function already done. Skipping.")
    else:
        reward_func = None

    # Train agent
    # env = get_env(self.env_name, reward_func)
    env = get_env(env_name, reward_func=None, difficulty=agent_config['difficulty'],
                  seed=agent_config.get('env_seed', None))  # provide reward_func via CustomPPO
    agent = RLAgent(env, agent_config, use_wandb=use_wandb, custom_reward_func=reward_func)

    agent.train()
    return agent