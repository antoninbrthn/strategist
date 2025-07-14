#!/usr/bin/env python
"""Evaluate the LLM policy agent on Crafter environments."""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import crafter

from strategist.io_utils import read_yaml_file
from strategist.llm_policy_agent import LLMPolicyAgent
from strategist.strategist_agent import get_env, get_goal_label

def run_episode(env, agent, max_steps=10000, render=False, verbose=False):
    """
    Run a single episode using the provided agent in the environment.
    
    Args:
        env: The environment to run the episode in.
        agent: The agent to use for action selection.
        max_steps: Maximum steps to run the episode for.
        render: Whether to render the environment.
        verbose: Whether to print verbose output.
        
    Returns:
        dict: The final trajectory information.
        float: The total reward accumulated during the episode.
    """
    obs = env.reset()
    done = False
    total_reward = 0.0
    step = 0
    trajectory = []
    
    # We'll use a default "noop" action for the first step
    # This allows us to get initial state information from the environment
    if verbose:
        print(f"Step {step}: Using default 'noop' action for first step")
    action_name = "noop"
    action_idx = 0  # noop is index 0
        
    # Take the initial action to get proper state information
    obs, reward, done, info = env.step(action_idx)
    info.update({"image": obs, "action": action_idx, "reward": reward, "done": done})
    total_reward += reward
    trajectory.append(info)
    step += 1
    
    # Main episode loop - now with proper state information
    while not done and step < max_steps:
        # Get action from agent using the current state info
        action_name = agent.get_action(info, verbose=verbose)
        action_idx = agent.convert_action_to_index(action_name)
        
        if verbose:
            print(f"Step {step}: Action {action_idx} ({action_name})")
        
        # Take action in environment
        obs, reward, done, info = env.step(action_idx)
        info.update({"image": obs, "action": action_idx, "reward": reward, "done": done})
        total_reward += reward
        
        trajectory.append(info)
        step += 1
        
        if render:
            env.render()
            time.sleep(0.1)  # Add delay to make rendering visible
    
    return trajectory, total_reward

def evaluate_agent(agent, env_name, difficulty, n_episodes=100, verbose=False, seed=None):
    """
    Evaluate the agent across multiple episodes.
    
    Args:
        agent: The agent to evaluate.
        env_name: The name of the environment.
        difficulty: The difficulty level of the environment.
        n_episodes: Number of episodes to evaluate for.
        verbose: Whether to print verbose output.
        
    Returns:
        dict: Statistics about the evaluation.
    """
    # Reset agent statistics
    agent.reset_stats()
    
    # Create environment
    env = get_env(env_name, difficulty=difficulty, seed=seed)
    
    # Run episodes
    rewards = []
    
    for episode in tqdm(range(n_episodes), desc="Evaluating"):
        trajectory, reward = run_episode(env, agent, verbose=verbose and episode == 0)
        rewards.append(reward)
        
        if verbose or episode % 10 == 0:
            print(f"Episode {episode+1}/{n_episodes}: Reward = {reward}")
    
    # Compile statistics
    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)
        
    results = {
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "success_rate": agent.get_success_rate(),
        "api_cost": agent.total_price,
        "all_rewards": rewards,
    }
    
    return results

def format_results(results):
    """Format results for printing."""
    output = [
        "Evaluation Results:",
        f"Mean Reward: {results['reward_mean']:.2f} ± {results['reward_std']:.2f}",
        f"LLM Action Success Rate: {results['success_rate']:.2%}",
        f"API Cost: ${results['api_cost']:.2f}",
    ]
    
    return output

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM policy agent on Crafter")
    parser.add_argument("--difficulty", type=str, choices=["easy", "medium", "surv"], default="easy",
                      help="Difficulty level of the Crafter environment")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", 
                      help="LLM model to use for policy decisions")
    parser.add_argument("--episodes", type=int, default=100,
                      help="Number of episodes to evaluate")
    parser.add_argument("--verbose", action="store_true",
                      help="Print verbose output")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set the seed for reproducibility
    np.random.seed(args.seed)
    
    # Load the goal from the appropriate YAML file
    if args.difficulty == "easy":
        config_file = "prompts/crafter-mod-farm-easy.yaml"
    elif args.difficulty == "medium":
        config_file = "prompts/crafter-mod-farm-medium.yaml"
    elif args.difficulty == "surv":
        raise NotImplementedError("Survival mode not yet supported")
    else:
        raise ValueError(f"Unknown difficulty: {args.difficulty}")
    
    # Read the YAML file
    config = read_yaml_file(config_file)
    goal = config["goal_context"].split("The overall goal is: ")[1].strip().strip('"')
    
    print(f"Evaluating LLM policy agent on Crafter {args.difficulty}")
    print(f"Goal: {goal}")
    print(f"Using model: {args.model}")
    print(f"Running {args.episodes} episodes\n")
    
    # Create LLM policy agent - now pass the difficulty parameter
    agent = LLMPolicyAgent(goal=goal, difficulty=args.difficulty, llm_model=args.model, use_azure=True)
    
    # Run evaluation
    results = evaluate_agent(
        agent=agent,
        env_name="crafter",
        difficulty=args.difficulty,
        n_episodes=args.episodes,
        verbose=args.verbose,
        seed=args.seed
    )
    
    # Print results
    print(format_results(results))
    
    # Save results to file
    output_dir = os.path.join("runs", f"llm_policy_{args.model}_{args.difficulty}")
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "results.json")
    import json
    with open(results_file, "w") as f:
        # Convert numpy values to Python types for JSON serialization
        serializable_results = {
            k: v if not isinstance(v, np.ndarray) and not isinstance(v, np.number) else v.tolist() 
            for k, v in results.items()
        }
        json.dump(serializable_results, f, indent=4)
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main() 