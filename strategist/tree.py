import math
import os
import json

import numpy as np


class Tree:
    def __init__(self, tree_path):
        self.tree_path = tree_path
        self.tree = []
        self._load_or_create_tree()

    def _load_or_create_tree(self):
        if os.path.exists(self.tree_path):
            self.tree = self._load_tree()
        else:
            pass  # don't create empty folder
            # self._save_tree()

    def _load_tree(self):
        with open(self.tree_path, "r") as file:
            return json.load(file)

    def _save_tree(self):
        os.makedirs(os.path.dirname(self.tree_path), exist_ok=True)
        with open(self.tree_path, "w") as file:
            json.dump(self.tree, file, indent=4)

    def init_tree(self, overall_goal=None):
        if overall_goal:
            self.tree = [
                {
                    "id": 1,
                    "goal": overall_goal,
                    "feasibility": 1,
                    "value": 1,
                    "parent": None,
                    "children": [],
                    "sequential": False,
                }
            ]
        else:
            self.tree = []
        self._save_tree()

    def apply_changes(self, llm_response):
        self.tree = apply_llm_changes(self.tree, llm_response)
        self._save_tree()

    def is_read_tree_command(self, llm_response):
        return "READ_TREE" in llm_response.upper()

    def get_tree_string(self):
        return "\n".join([str(node) for node in self.tree])

    def find_most_promising_leaf(self, start_node_id=None):
        def is_leaf(node):
            return not node["children"]

        def calculate_promise(node):
            return node["feasibility"] * node["value"]

        def dfs(node_id):
            node = find_node_by_id(self.tree, node_id)
            if is_leaf(node):
                return node_id, calculate_promise(node)

            if node["sequential"]:
                return node_id, calculate_promise(node)

            best_leaf_id = None
            best_promise = float("-inf")
            for child_id in node["children"]:
                leaf_id, promise = dfs(child_id)
                if promise > best_promise:
                    best_leaf_id = leaf_id
                    best_promise = promise

            return best_leaf_id, best_promise

        start_node_id = start_node_id or self.tree[0]["id"]  # Use root if not specified
        most_promising_leaf_id, _ = dfs(start_node_id)
        return most_promising_leaf_id

    def extract_root_to_leaf_path(self, node_id, root_id=None):
        path = []
        current_id = node_id

        # First, collect all parent nodes up to the root
        while (current_id is not None) and (current_id != root_id):
            node = find_node_by_id(self.tree, current_id)
            path.append(node)
            current_id = node["parent"]

        path = list(reversed(path))  # Reverse to get root-to-leaf order

        # Now handle sequential nodes
        for i, node in enumerate(path):
            if node["sequential"]:
                seq_children_paths = []
                for child_id in node["children"]:
                    child_leaf_id = self.find_most_promising_leaf(child_id)
                    child_path = self.extract_root_to_leaf_path(
                        child_leaf_id, root_id=node_id
                    )
                    seq_children_paths.extend(
                        child_path[1:]
                    )  # Exclude the sequential node itself

                # Replace the sequential node and its descendants with the new paths
                path = path[: i + 1] + seq_children_paths

        return path

    def extract_all_leaf_paths(self, stop_at_plan=True):
        """Extract all paths from root to leaf nodes. If `stop_at_plan`, stop at sequential nodes."""
        paths = []

        def dfs(node_id, current_path):
            node = find_node_by_id(self.tree, node_id)
            current_path.append(node)

            if not node["children"] or (stop_at_plan and node["sequential"]):
                paths.append(current_path)
                return

            for child_id in node["children"]:
                dfs(child_id, current_path.copy())

        dfs(self.tree[0]["id"], [])

        return paths


# Find node by id
def find_node_by_id(tree, node_id):
    for node in tree:
        if node["id"] == node_id:
            return node
    return None


def update_node(tree, node_id, new_feasibility, new_value):
    node = find_node_by_id(tree, node_id)
    if node:
        node["feasibility"] = new_feasibility
        node["value"] = new_value
    return tree


def add_node(
    tree, parent_id, new_goal, new_feasibility=None, new_value=None, sequential=False
):
    new_id = max(node["id"] for node in tree) + 1

    parent_node = find_node_by_id(tree, parent_id)
    parent_node["children"].append(new_id)

    new_node = {
        "id": new_id,
        "goal": new_goal,
        "feasibility": new_feasibility,
        "value": new_value,
        "parent": parent_id,
        "children": [],
        "sequential": sequential,
    }
    tree.append(new_node)

    return tree


def add_sequential_node(tree, plan_node_id, new_goals, feasibilities, values):
    assert len(feasibilities) == len(new_goals)
    assert len(values) == len(new_goals)

    plan_node = find_node_by_id(tree, plan_node_id)
    if not plan_node:
        print(f"Warning: Parent node with ID {plan_node_id} not found.")
        return tree

    new_ids = []
    for i, goal in enumerate(new_goals):
        new_id = max(node["id"] for node in tree) + 1
        new_ids.append(new_id)

        plan_node["children"].append(new_id)

        new_node = {
            "id": new_id,
            "goal": goal,
            "feasibility": feasibilities[i],
            "value": values[i],
            "parent": plan_node_id,
            "children": [],
            "sequential": False,
        }
        tree.append(new_node)
    return tree


def update_sequential_node_metrics(tree, seq_node_id):
    seq_node = find_node_by_id(tree, seq_node_id)
    if seq_node and seq_node["sequential"]:
        child_nodes = [
            find_node_by_id(tree, child_id) for child_id in seq_node["children"]
        ]
        # feasibility = min(feasibility across children)
        seq_node["feasibility"] = min(
            node["feasibility"]
            for node in child_nodes
            if node["feasibility"] is not None
        )
        # value of plan = average of values under plan
        seq_node["value"] = np.mean(
            [node["value"] for node in child_nodes if node["value"] is not None]
        )
    return tree


def remove_node(tree, node_id):
    node_to_remove = find_node_by_id(tree, node_id)
    if not node_to_remove:
        print("No nodes to remove")
        return tree

    parent_id = node_to_remove["parent"]
    parent_node = find_node_by_id(tree, parent_id)

    if parent_node:
        for i, child in enumerate(parent_node["children"]):
            if isinstance(child, tuple):
                if node_id in child:
                    if len(child) > 1:
                        parent_node["children"][i] = tuple(
                            id for id in child if id != node_id
                        )
                    else:
                        parent_node["children"].pop(i)
                    break
            elif child == node_id:
                parent_node["children"].pop(i)
                break

    def remove_children(node_id):
        node = find_node_by_id(tree, node_id)
        if not node:
            return

        for child in node["children"]:
            if isinstance(child, tuple):
                for child_id in child:
                    remove_children(child_id)
            else:
                remove_children(child)

        tree.remove(node)

    remove_children(node_id)
    return tree


# Check validity of the tree
def is_tree_valid(tree):
    # Check for unique ids
    ids = [node["id"] for node in tree]
    if len(ids) != len(set(ids)):
        print("Error: Duplicate node IDs detected.")
        return False

    # Find the root node (it has no parent)
    root_nodes = [node for node in tree if node["parent"] is None]
    if len(root_nodes) != 1:
        print(f"Error: There should be exactly one root node, found {len(root_nodes)}.")
        return False

    # Check for cycles and correct parent-child relationships
    def check_node(node_id, visited):
        if node_id in visited:
            return False  # Cycle detected
        visited.add(node_id)
        node = find_node_by_id(tree, node_id)
        if not node:
            return False  # Node not found (invalid reference)

        for child_id in node["children"]:
            child = find_node_by_id(tree, child_id)
            if not child or child["parent"] != node_id:
                print(
                    f"Error: Child {child_id} does not correctly reference its parent {node_id}."
                )
                return False

            if not check_node(child_id, visited):
                return False
        return True

    # Traverse from the root and check the whole tree
    visited = set()
    root_node = root_nodes[0]
    if not check_node(root_node["id"], visited):
        print("Error: Cycle detected or parent-child relationship error.")
        return False

    # Ensure all nodes were visited (i.e., no orphaned nodes)
    if len(visited) != len(tree):
        print("Error: Not all nodes are reachable from the root.")
        return False

    return True


def is_read_tree_command(llm_output):
    return "READ_TREE" in llm_output


def apply_llm_changes(tree, llm_output):
    lines = llm_output.strip().split("\n")
    for line in lines:
        if line.startswith("ADD|"):
            parts = line.split("|")
            if len(parts) == 5:
                goal = parts[1]
                parent_id = parts[2]
                feasibility = float(parts[3])
                value = float(parts[4])
                tree = add_node(tree, int(parent_id), goal, feasibility, value)
            else:
                raise ValueError

        elif line.startswith("ADD_SEQUENTIAL|"):
            parts = line.split("|")
            if len(parts) >= 5:  # Ensure there are enough parts
                goals = (
                    parts[1].strip('"').split(";")
                )  # Strip quotes and split by semicolon
                parent_id = parts[2]

                # Prepare feasibility and value lists
                feasibilities = []
                values = []
                for i in range(3, len(parts)):
                    if i % 2 == 0:  # Even indices are feasibilities
                        feasibilities.append(float(parts[i]))
                    else:  # Odd indices are values
                        values.append(float(parts[i]))

                plan_goal = f"Plan({';'.join(goals)})"
                # add goal
                tree = add_node(
                    tree, int(parent_id), plan_goal, None, None, sequential=True
                )
                plan_id = max(node["id"] for node in tree)
                tree = add_sequential_node(
                    tree, int(plan_id), goals, feasibilities, values
                )

                tree = update_sequential_node_metrics(tree, plan_id)
            else:
                raise ValueError

        elif line.startswith("REMOVE|"):
            _, node_id = line.split("|")
            tree = remove_node(tree, int(node_id))

        elif line.startswith("MODIFY|"):
            parts = line.split("|")
            if len(parts) == 4:
                node_id = parts[1]
                new_feasibility = float(parts[2])
                new_value = float(parts[3])
                tree = update_node(tree, int(node_id), new_feasibility, new_value)

    return tree
