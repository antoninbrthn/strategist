import os
import pickle
from typing import Dict
from functools import wraps

import yaml
import json


def read_yaml_file(filepath):
    """Utility function to read a YAML file."""
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def read_json_file(filepath):
    """Utility function to read a JSON file."""
    with open(filepath, "r") as file:
        return json.load(file)


def read_text_file(filepath):
    """Utility function to read a text file."""
    with open(filepath, "r") as file:
        return file.read()


def load_prompt(file_path: str) -> Dict[str, str]:
    """Load prompts from a YAML file."""
    with open(file_path, "r") as file:
        prompt_data = yaml.safe_load(file)
    return prompt_data


def create_prompt_from_list(prompts):
    """Combine a list of prompts into a single prompt."""
    return "\n".join(prompts)


def ensure_directory_exists(directory):
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)


# Decorator for checkpoint management
def checkpoint_pkl(export_dir, default_filename=None):
    """
    A decorator to manage computation/export checkpoints.

    Args:
        export_dir (str): Directory to save/load checkpoints.
        default_filename (str): Default filename for the checkpoint.

    Returns:
        The decorator.

    Example usage:
    - `func(*, checkpoint_file="custom_name.pkl")`: Uses a custom filename for the checkpoint.
    - `func(*, use_checkpoint=False)`: Bypasses the checkpoint and recomputes the result.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, checkpoint_file=None, use_checkpoint=True, **kwargs):
            # Ensure the export directory exists
            os.makedirs(export_dir, exist_ok=True)

            # Determine the file path
            if checkpoint_file:
                path = os.path.join(export_dir, checkpoint_file)
            else:
                # Auto-generate filename if not provided
                fn = default_filename or func.__name__
                i = 0
                while os.path.exists(os.path.join(export_dir, f"{fn}_{i}.pkl")):
                    i += 1
                path = os.path.join(export_dir, f"{fn}_{i}.pkl")

            # Load from checkpoint if available
            if use_checkpoint and os.path.exists(path):
                print(f"Loading from checkpoint: {path}")
                with open(path, "rb") as f:
                    return pickle.load(f)

            # Compute the result
            print(f"Checkpoint not found or bypassed. Computing {func.__name__}...")
            result = func(*args, **kwargs)

            # Save the result to the checkpoint
            print(f"Saving to checkpoint: {path}")
            with open(path, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator

