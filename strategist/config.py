import os

from strategist.io_utils import read_yaml_file, load_prompt

PROMPTS_DIR = "./prompts/"
RUNS_DIR = "./runs/"
CONFIGS_DIR = "./configs/"
EXPORT_DIR = "./export/"
FIGS_DIR = "./results/figs/"
TABLES_DIR = "./results/tables/"

def load_config(config_file):
    """Load a config file from the CONFIGS_DIR"""
    config_path = os.path.join(CONFIGS_DIR, config_file)
    config = read_yaml_file(config_path)
    return config

class Config:
    def __init__(self, config_file="test_config.yaml", project_root=None):
        self.project_root = project_root or os.getcwd()
        config = self._load_config(config_file)

        self.runs_dir = os.path.join(self.project_root, RUNS_DIR)
        self.model = config["model"]
        self.alias = config["alias"]
        self.base_prompt_file = os.path.join(self.project_root, PROMPTS_DIR, config["base_prompt_file"])
        self.game_prompt_file = os.path.join(self.project_root, PROMPTS_DIR, config["game_prompt_file"])
        self.run_id = self._get_run_id()
        self.env_name = config.get("env_name", None)

    def _load_config(self, config_file):
        config_path = os.path.join(self.project_root, CONFIGS_DIR, config_file)
        return read_yaml_file(config_path)

    @property
    def tree_file(self):
        return os.path.join(self.current_run_dir, "tree.json")

    @property
    def convo_file(self):
        return os.path.join(self.current_run_dir, "convo.json")

    @property
    def current_run_dir(self):
        return os.path.join(self.runs_dir, self.run_id)

    def _get_run_id(self):
        base_run_id = f"{self.alias}_{self.model}"
        run_id = base_run_id
        i = 0
        while os.path.exists(os.path.join(self.runs_dir, run_id)):
            run_id = f"{base_run_id}_{i}"
            i += 1
        return run_id

    def read_base_prompt(self):
        return load_prompt(self.base_prompt_file)

    def read_game_prompt(self):
        return load_prompt(self.game_prompt_file)

    def read_combined_prompts(self):
        base_prompts = self.read_base_prompt()
        game_prompts = self.read_game_prompt()
        combined_prompts = {**base_prompts, **game_prompts}
        return combined_prompts
