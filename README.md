# Strategic Planning: A Top-Down Approach To Option Generation

Code for "Strategic Planning: A Top-Down Approach To Option Generation", Ruiz Lutyten et al (2024).

## Getting Started

### 1. Environment Setup

First, create and activate the conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate strategist
```

### 1b. Initialize Submodules

This repository uses several git submodules (see `.gitmodules`). To ensure all dependencies are available, initialize and update the submodules after cloning:

```bash
git submodule update --init --recursive
```

This will fetch the following submodules:
- `data/crafter` (https://github.com/antoninbrthn/crafter.git)
- `data/SmartPlay` (https://github.com/maxruizluyten/SmartPlay.git)
- `Eureka` (https://github.com/maxruizluyten/Eureka.git)

### 1c. Install Custom Crafter Environments

To use the custom Crafter environments required for this project, follow these steps:

1. **Clone the forked Crafter repository:**
   ```bash
   git clone https://github.com/antoninbrthn/crafter.git
   cd crafter
   ```
2. **Select the appropriate branch:**
   - For Crafter-Easy and Crafter-Medium (farming mod):
     ```bash
     git checkout farming-mod
     ```
   - For the original Crafter environment:
     ```bash
     git checkout og
     ```
3. **Install the package in editable mode:**
   ```bash
   pip install -e .
   ```
4. **Check the installation:**
   - The installed package version should be:
     - `1.8.3+farm` for the farming-mod branch
     - `1.8.3+og` for the og branch
   - You can check this by running:
     ```bash
     python -c "import crafter; print(crafter.__version__)"
     ```

### 2. Running Experiments

The main workflow for creating a strategy tree and training an agent is handled by `scripts/run_full_strategist.py`. The process is split into two steps:

#### Step 1: Create the Strategy Tree

Run this first to generate and save the strategy tree for your chosen environment:

```bash
python scripts/run_full_strategist.py --difficulty <difficulty> --make_tree True
```
- `--make_tree`: If set to `True`, only creates and saves the strategy tree for the specified environment. No agent training is performed in this step.

#### Step 2: Train an Agent Using a Strategy Tree

Once the tree is created, you can train an agent on a specific branch or node:

```bash
python scripts/run_full_strategist.py --difficulty <difficulty> --branch_i <branch_index> --config <config_file> [other options]
```
- `--run_id <str>`: (Optional) Specify the run ID to load a particular strategy tree. By default, this is set to the strategy trees used in the paper for each environment (`crafter-mod-farm-easy_gpt-4o`, `crafter-mod-farm-medium_gpt-4o`, etc.), so you only need to provide this if you want to use a custom or previously generated tree.

#### Main Command-Line Arguments

- `--config <file>`: Path to the agent config YAML (e.g., `configs/rl_agent.yaml`, `configs/rl_agent-ppo.yaml`). This controls the RL algorithm, hyperparameters, and reward model settings.
- `--difficulty <easy|medium|og>`: Selects the environment difficulty. This also determines which environment config is loaded (see `configs/`).
- `--make_tree <bool>`: If `True`, only creates and saves the strategy tree for the specified environment.
- `--run_id <str>`: Run ID to load a specific strategy tree. Defaults to the strategy trees used in the paper for each environment.
- `--branch_i <int>`: Index of the strategy tree branch to train on. Use this to select a specific subgoal/plan.
- `--node <int>`: (Alternative to `branch_i`) Specify a node id directly.
- `--alpha <float>`: Sets the alpha parameter for reward shaping (overrides config).
- `--alpha_schedule <start> <end> <decay_start> <decay_end>`: Schedule for alpha decay (overrides config).
- `--decay_function <str>`: Shape of alpha decay (e.g., `linear`, `poly_0.5`).
- `--reward_mode <str>`: Reward shaping mode (e.g., `direct`, `diff`, `only_custom`).
- `--train_ts <int>`: Number of training timesteps (overrides config).
- `--log_interval <int>`: Logging interval for wandb (overrides config).
- `--wandb_tags <tag1> <tag2> ...`: Tags for experiment tracking in Weights & Biases.
- `--seed <int>`: Random seed for reproducibility.
- `--verbose <bool>`: Verbose output.

#### Example Workflow

1. **Create the strategy tree:**
   ```bash
   python scripts/run_full_strategist.py --difficulty medium --make_tree True
   ```
2. **Train an agent on a branch of the tree:**
   ```bash
   python scripts/run_full_strategist.py --difficulty medium --branch_i 0 --config configs/rl_agent.yaml --alpha_schedule 1 0 50_000 1_000_000 --train_ts 2_000_000
   ```
   *(You can add `--run_id <your_run_id>` if you want to use a custom tree.)*

### 3. Configurations

- **Agent configs**: Located in `configs/` (e.g., `rl_agent.yaml`, `rl_agent-ppo.yaml`). These files control the RL algorithm, model parameters, training length, and reward model settings.
- **Environment configs**: Also in `configs/`, named like `crafter-mod-farm-easy_gpt-4o.yaml`, etc. These are selected automatically based on the `--difficulty` argument.
- **Prompts**: All prompt files are in the `prompts/` directory. For example, `prompts/crafter-mod-farm-easy.yaml` contains the context and instructions for the easy environment. Reward shaping prompt templates are in `prompts/reward_shaper/`.

To modify the agent's behavior, edit the relevant config YAML in `configs/`. To change the environment description or reward shaping instructions, edit the appropriate YAML or text file in `prompts/`.

## LLM Policy Evaluation

To evaluate the performance of the LLM policy agent directly on the Crafter environment, you can use the `scripts/evaluate_llm_policy.py` script. This script runs the agent for a specified number of episodes and reports the average reward and other metrics.

### Usage

To run the evaluation, use the following command structure:

```bash
python scripts/evaluate_llm_policy.py [OPTIONS]
```

### Arguments

-   `--difficulty`: The difficulty level of the Crafter environment. Choose from `easy`, `medium`. The default is `easy`.
-   `--model`: The LLM model to be used for policy decisions (e.g., `gpt-4o-mini`, `gpt-4`). The default is `gpt-4o-mini`.
-   `--episodes`: The number of episodes to run the evaluation for. The default is `100`.
-   `--verbose`: If included, it prints detailed logs of the agent's actions and environment states.
-   `--seed`: An integer to set the random seed for reproducibility. The default is `42`.

### Example

Here is an example command to run the evaluation on the `easy` difficulty with the `gpt-4o-mini` model for 10 episodes:

```bash
python scripts/evaluate_llm_policy.py --difficulty easy --model gpt-4o-mini --episodes 10 --seed 42
```

The results of the evaluation, including mean reward, success rate, and total API cost, will be printed to the console and saved in a JSON file under the `runs/llm_policy_[MODEL]_[DIFFICULTY]` directory.

## Eureka vs Strategist Comparison

This branch includes tools for running Eureka on the Crafter environment.

Before running Eureka experiments, make sure you are on the correct branch. Switch to the `Eureka` branch using:
```
git checkout Eureka --
```

To run the Eureka experiment, use the following command:

```bash
python Eureka/eureka/eureka.py env=crafter-mod
```

The results of the evaluation, including mean reward, success rate, and total API cost, will be printed to the console and saved in a JSON file under the `runs/eureka_[MODEL]_[DIFFICULTY]` directory.