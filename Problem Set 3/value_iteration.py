from typing import Dict, Optional
from agents import Agent
from environment import Environment
from mdp import MarkovDecisionProcess, S, A
import json

from helpers.utils import NotImplemented


# This is a class for a generic Value Iteration agent
class ValueIterationAgent(Agent[S, A]):
    mdp: MarkovDecisionProcess[S, A]  # The MDP used by this agent for training
    utilities: Dict[S, float]  # The computed utilities
    # The key is the string representation of the state and the value is the utility
    discount_factor: float  # The discount factor (gamma)

    def __init__(self, mdp: MarkovDecisionProcess[S, A], discount_factor: float = 0.99) -> None:
        super().__init__()
        self.mdp = mdp
        self.utilities = {state: 0 for state in self.mdp.get_states()}  # We initialize all the utilities to be 0
        self.discount_factor = discount_factor

    # Given a state, compute its utility using the bellman equation
    # if the state is terminal, return 0
    def compute_bellman(self, state: S) -> float:
        """
        :param state:
        :return: utility of the state
        """
        """
        apply bellman equation to get the utility of this state, but with modification 
        as the reward is fuction of the current state, the action, and the next state
        """
        values = []
        if self.mdp.is_terminal(state):
            return 0
        actions = self.mdp.get_actions(state)
        for action in actions:
            successors = self.mdp.get_successor(state, action)
            value = 0
            for successor, probability in successors.items():
                reward = self.mdp.get_reward(state, action, successor)
                value += probability * (reward + (self.discount_factor * self.utilities[successor]))
            values.append(value)
        values.sort()
        return values[-1]

    # Applies a single utility update
    # then returns True if the utilities has converged (the maximum utility change is less or equal the tolerance)
    # and False otherwise
    def update(self, tolerance: float = 0) -> bool:
        """
        :param tolerance:
        :return: True if the utilities has converged
        """
        """
        compute new utilities using bellman equation, then check if the difference between the new
        utility and the old utility for each state and, if the difference is bigger than the tolerance
        then we finally return false after modifiying the utilities 
        """
        newUtilities = {}
        flag = True
        for state in self.mdp.get_states():
            newUtilities[state] = self.compute_bellman(state)
        for state in self.mdp.get_states():
            if abs(self.utilities[state] - newUtilities[state]) >= tolerance:
                flag = False
        self.utilities = newUtilities
        return flag

    # This function applies value iteration starting from the current utilities stored in the agent and stores the new utilities in the agent
    # NOTE: this function does incremental update and does not clear the utilities to 0 before running
    # In other words, calling train(M) followed by train(N) is equivalent to just calling train(N+M)
    def train(self, iterations: Optional[int] = None, tolerance: float = 0) -> int:
        iteration = 0
        while iterations is None or iteration < iterations:
            iteration += 1
            if self.update(tolerance):
                break
        return iteration

    # Given an environment and a state, return the best action as guided by the learned utilities and the MDP
    # If the state is terminal, return None
    def act(self, env: Environment[S, A], state: S) -> A:
        """
        :param env:
        :param state:
        :return: the action that led to that utility
        """
        """
        similar to compute_bellman but modifying the return to be the action instead of the value itself
        """
        values = []
        if self.mdp.is_terminal(state):
            return 0
        actions = env.actions()
        for action in actions:
            successors = self.mdp.get_successor(state, action)
            value = 0
            for successor, probability in successors.items():
                reward = self.mdp.get_reward(state, action, successor)
                value += probability * (reward + (self.discount_factor * self.utilities[successor]))
            values.append((value, action))
        values.sort(key=lambda item: -item[0])
        return values[0][1]

    # Save the utilities to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            utilities = {self.mdp.format_state(state): value for state, value in self.utilities.items()}
            json.dump(utilities, f, indent=2, sort_keys=True)

    # loads the utilities from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            utilities = json.load(f)
            self.utilities = {self.mdp.parse_state(state): value for state, value in utilities.items()}
