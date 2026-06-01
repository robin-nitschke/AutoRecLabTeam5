import pickle
import random
from pathlib import Path

from anytree import PreOrderIter

from config import Config

# from treesearch.minimal_agent import MinimalAgent
from treesearch.interpreter import Interpreter
from treesearch.minimal_agent import MinimalAgent
from treesearch.node import Node
from utils.log import _ROOT_LOGGER

logger = _ROOT_LOGGER.getChild("treesearch")


class TreeSearch:
    def __init__(self, user_request: str, config: Config) -> None:
        self._user_request = user_request
        self._config = config
        self._draft_nodes: list[Node] = []
        workspace_pth = Path(config.exec.workspace).resolve()
        workspace_pth.mkdir(exist_ok=True, parents=True)
        self._workspace = str(workspace_pth)

        self._minimal_agent = MinimalAgent(self._task_desc, self._config)
        self._interpreter = Interpreter(self._workspace, self._config.exec.timeout)

    @property
    def all_nodes(self):
        return [n for root in self._draft_nodes for n in PreOrderIter(root)]

    @property
    def good_nodes(self):
        return list(filter(lambda n: not n.is_buggy, self.all_nodes))

    @property
    def buggy_nodes(self):
        return list(filter(lambda n: n.is_buggy, self.all_nodes))

    @property
    def best_good_node(self):
        # Fall back to the highest-scoring node overall if no node was marked
        # satisfactory — otherwise the final summary crashes with IndexError.
        candidates = self.good_nodes or self.all_nodes
        if not candidates:
            return None
        candidates.sort(key=lambda n: n.score.score, reverse=True)
        return candidates[0]

    def select_next_node(self) -> Node:
        if (
            len(self.buggy_nodes) > 0
            and random.random() < self._config.treesearch.debug_prob
            or len(self.good_nodes) == 0
        ):
            return random.choice(self.buggy_nodes)

        # Epsilon-greedy explore vs. exploit:
        if random.random() < self._config.treesearch.epsilon:
            return random.choice(self.good_nodes)
        else:
            return self.best_good_node

    def run(self):
        logger.info("Starting tree search...")
        # Step 1: Generate draft nodes:
        for i in range(self._config.treesearch.num_draft_nodes):
            logger.info(
                f"Generating draft node {i + 1}/{self._config.treesearch.num_draft_nodes}"
            )
            draft_node = self._minimal_agent._draft()
            self.exec_node(draft_node)
            self._draft_nodes.append(draft_node)

        for i in range(self._config.treesearch.max_iterations):
            logger.info(
                f"Treesearch iteration {i + 1}/{self._config.treesearch.max_iterations}"
            )
            parent_node = self.select_next_node()

            if parent_node.is_buggy:
                child_node = self._minimal_agent._debug(parent_node)
            else:
                child_node = self._minimal_agent._improve(parent_node)

            self.exec_node(child_node)

            if child_node.score.is_satisfactory:
                logger.info("Found satisfactory node:")
                self.save()
                self.print_experiment_summary(child_node)
                return

        self.save()

        logger.warning("Found no satisfactory node; Using best node instead...")

        best_node = self.best_good_node
        if best_node is None:
            logger.error("No nodes were produced — cannot print experiment summary.")
            return
        self.print_experiment_summary(best_node)

    def exec_node(self, node: Node) -> Node:
        exec_result = self._interpreter.run(node.code)
        logger.debug(exec_result)
        self._minimal_agent.score_code(node, exec_result)
        return node

    def print_experiment_summary(self, result_node: Node):
        logger.info("Final response:")
        print(self._minimal_agent._summarize(self._user_request, result_node))

    @property
    def _task_desc(self) -> str:
        task_desc = """ You are an expert recommender systems research assistant who is looking to help the user with their requests.
The user has some idea and you want to conduct creative experiments to gain scientific insights.
Your aim is to run experiments to gather sufficient results to report back to the user.
The idea is:\n
"""
        task_desc += self._user_request
        return task_desc

    def save(self):
        with open("./out/save.pkl", "wb") as f:
            logger.warning(f"SAVING {len(self._draft_nodes)}.....")
            pickle.dump(self._draft_nodes, f)
