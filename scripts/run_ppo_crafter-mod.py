"""Train PPO on Crafter-mod environment."""
import os

import crafter
import sys

from strategist.config import load_config
from strategist.rl_agent import RLAgent

config = load_config('rl_agent-ppo.yaml')

run_checkpoints = {  
                # 1: ("strategist/export/crafter_PPO_20241203-180300", "2izwinic"),  # example checkpoint
                   }

# get checkpoint from command line arg
if len(sys.argv) > 1:
    load_checkpoint, run_id = run_checkpoints[int(sys.argv[1])]
    checkpoints = os.listdir(os.path.join(load_checkpoint, "checkpoints"))
    checkpoints = [int(c.split("_")[1].split(".")[0]) for c in checkpoints]
    latest_checkpoint = max(checkpoints)
    checkpoint_file = os.path.join(load_checkpoint, "checkpoints",
                                    f"checkpoint_{latest_checkpoint}.zip")
    n_runs = 1
else:
    checkpoint_file = None
    run_id = None
    latest_checkpoint = None
    n_runs = 3

for run in range(n_runs):
    print(f"Run {run+1}/{n_runs}")
    env = crafter.Env()
    agent = RLAgent(env, config, use_wandb=True, run_id=run_id)
    if checkpoint_file:
        agent.load_from_checkpoint(checkpoint_file)
        print('Loaded checkpoint:', checkpoint_file)
        agent.train_ts = agent.train_ts - latest_checkpoint
    print('Training agent for ', agent.train_ts, 'timesteps')
    agent.train()



