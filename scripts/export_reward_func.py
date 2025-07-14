"""
Export reward function from a trained model in a way where it can easily be loaded without access to the Strategist package.

Run from the root of the Strategist repo.
"""
import os
import torch
import sys
from strategist.reward_shaper_text_score import TextRewardShaperScore

args = {'strategy': 'hunt'}
reward_func_paths = {
    # 'og-fight': 'strategist/export/annotations/<path to reward_model.pkl>' # Example path to a strategy's reward model.
}

for strat, path in reward_func_paths.items():
    assert os.path.exists(path), f"Path does not exist: {path}"
    reward_func_path = path
    reward_model = torch.load(reward_func_path)
    example_input = torch.randn(1, 3, 64, 64)
    traced_model = torch.jit.trace(reward_model, example_input)

    # Save the traced model
    export_path = os.path.join(os.path.dirname(reward_func_path), "reward_model_general.pt")
    traced_model.save(export_path)
    print("Model saved to", export_path)

