import argparse
import sys

from strategist.rl_agent import set_global_seed
from strategist.config import load_config
from strategist.strategist_agent import Strategist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, help="Path to config YAML file", default="rl_agent.yaml")
    parser.add_argument('--node', type=int, required=False, help="Node index to run from", default=None)
    parser.add_argument('--branch_i', type=int, required=False, help="Branch index to run from", default=None)
    parser.add_argument('--difficulty', type=str, required=False, help="Farm-mod difficulty", default='easy')
    parser.add_argument('--alpha', type=int, required=False, help="Alpha parameter", default=None)
    parser.add_argument('--alpha_schedule', type=float, nargs=4, required=False, help="Alpha scheduling parameters", default=None)
    parser.add_argument('--decay_function', type=str, required=False, help="Alpha decay shape", default=None)
    parser.add_argument('--reward_mode', type=str, required=False, help="Custom reward mode", default=None)
    parser.add_argument('--train_ts', type=int, required=False, help="Timesteps for training", default=None)
    parser.add_argument('--log_interval', type=int, required=False, help="Intervals for logging results to wandb", default=None)
    parser.add_argument('--wandb_tags', type=str, nargs="+", required=False, help="WandB tags", default=[])
    parser.add_argument('--verbose', type=bool, required=False, help="Verbose mode", default=False)
    parser.add_argument('--seed', type=int, default=42, help="Global random seed")
    parser.add_argument('--make_tree', type=bool, default=False, help="Make the initial Strategy Tree")
    parser.add_argument('--run_id', type=str, default=None, help="Run ID to load the Strategy Tree from")

    is_console = "pydevconsole.py" in sys.argv[0]
    args = parser.parse_args() if not is_console else parser.parse_args([])
    set_global_seed(args.seed)
    print("Set global seed to ", args.seed)

    # Create Strategy Tree
    if args.difficulty == 'easy':
        config_file = "crafter-mod-farm-easy_gpt-4o.yaml"
    elif args.difficulty == 'medium':
        config_file = "crafter-mod-farm-medium_gpt-4o.yaml"
    elif args.difficulty == 'surv':
        config_file = "crafter-surv_gpt-4o.yaml"
    elif args.difficulty == 'og':
        config_file = "crafter_gpt-4o.yaml"
    else:
        raise ValueError(f"Unknown difficulty: {args.difficulty}")
    strategist = Strategist(max_rounds=10, config_file=config_file, use_azure=True)
    if args.make_tree:
        print('Creating Strategy Tree...')
        strategist.run()  # run this first, then run the rest from checkpoint
        # display the final tree
        print('Final Strategy Tree')
        print(strategist.tree.get_tree_string())
        exit()

    if args.run_id is not None:
        run_id = args.run_id
    elif args.difficulty == 'easy':
        run_id = "crafter-mod-farm-easy_gpt-4o"
    elif args.difficulty == 'medium':
        run_id = "crafter-mod-farm-medium_gpt-4o"
    elif args.difficulty == 'og':
        run_id = "crafter_gpt-4o"
    else:
        raise ValueError(f"Unknown difficulty: {args.difficulty}")

    strategist.load_run(run_id)
    all_branches = strategist.tree.extract_all_leaf_paths(stop_at_plan=True)
    print(strategist.tree.get_tree_string())
    for branch in all_branches:
        print(f"Id {branch[-1]['id']}: {branch[-1]['goal']}")
    node_ids = [node[-1]['id'] for node in all_branches]

    if args.branch_i is not None:
        node_id = node_ids[args.branch_i]
    elif args.node is not None:
        node_id = args.node
    else:
        node_id = None
    print('Node id:', node_id)

    # Init RL agent
    agent_config = load_config(args.config)
    agent_config["difficulty"] = args.difficulty
    agent_config["run_id"] = run_id
    agent_config["node_id"] = node_id
    agent_config["wandb_tags"] = args.wandb_tags
    if args.alpha is not None:
        agent_config["model_params"]["alpha"] = args.alpha
    if args.reward_mode is not None:
        agent_config["model_params"]["reward_mode"] = args.reward_mode
    if args.alpha_schedule is not None:
        agent_config["model_params"]["alpha_schedule"] = args.alpha_schedule
        print(f"Using alpha schedule: {args.alpha_schedule}")
    if args.decay_function is not None:
        agent_config["model_params"]["decay_function"] = args.decay_function
        print(f"Using decay function: {args.decay_function}")
    if args.train_ts is not None:
        agent_config["train_ts"] = args.train_ts
    if args.log_interval is not None:
        agent_config["log_interval"] = args.log_interval
    if args.seed is not None:
        agent_config["env_seed"] = args.seed
    strategist.agent_config = agent_config

    # Train an agent on the node given the Strategy Tree
    agent = strategist.do_node_v2(node_id=node_id, verbose=args.verbose)



