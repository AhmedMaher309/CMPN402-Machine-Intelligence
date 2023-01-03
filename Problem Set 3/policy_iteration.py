from typing import Dict, Optional
from agents import Agent
from environment import Environment
from mdp import MarkovDecisionProcess, S, A
import json
import numpy as np
import copy

from helpers.utils import NotImplemented

# This is a class for a generic Policy Iteration agent
class PolicyIterationAgent(Agent[S, A]):
    mdp: MarkovDecisionProcess[S, A] # The MDP used by this agent for training
    policy: Dict[S, A]
    utilities: Dict[S, float] # The computed utilities
                                # The key is the string representation of the state and the value is the utility
    discount_factor: float # The discount factor (gamma)

    def __init__(self, mdp: MarkovDecisionProcess[S, A], discount_factor: float = 0.99) -> None:
        super().__init__()
        self.mdp = mdp
        # This initial policy will contain the first available action for each state,
        # except for terminal states where the policy should return None.
        self.policy = {
            state: (None if self.mdp.is_terminal(state) else self.mdp.get_actions(state)[0])
            for state in self.mdp.get_states()
        }
        self.utilities = {state:0 for state in self.mdp.get_states()} # We initialize all the utilities to be 0
        self.discount_factor = discount_factor

    # Given the utilities for the current policy, compute the new policy
    def update_policy(self):
        """
        compute the policy for each state and modify policies according to it.
        we loop on states and compute the utility for each state and the policy is the action led to that utility
        """
        for state in self.mdp.get_states():
            if self.mdp.is_terminal(state):
                continue
            bestValue = -99999999
            bestAction = None
            for action in self.mdp.get_actions(state):
                successors = self.mdp.get_successor(state, action)
                value = 0
                for successor, probability in successors.items():
                    reward = self.mdp.get_reward(state, action, successor)
                    value += probability * (reward + (self.discount_factor * self.utilities[successor]))
                if value > bestValue:
                    bestValue = value
                    bestAction = action
            self.policy[state] = bestAction

    # Given the current policy, compute the utilities for this policy
    # Hint: you can use numpy to solve the linear equations. We recommend that you use numpy.linalg.lstsq
    def update_utilities(self):
        """
        According to AX = B such that both A, b are matrices, we create these to matrices
        from the equation of policy evaluation in order to compute and update utilities
        using numpy to solve the equation
        using coefficients matrix for the left hand side and the other one is Right hand side as it is named
        then update all utilities with the new calculated ones from the equation
        """
        coefficients = np.identity(len(self.mdp.get_states()))
        RightHandSide = np.zeros(len(self.mdp.get_states()))
        indexDict = {}
        index: int = 0
        for state in self.mdp.get_states():
            indexDict[state] = index
            index += 1
        for state in self.mdp.get_states():
            if not self.mdp.is_terminal(state):
                action = self.policy[state]
                for successor, probability in self.mdp.get_successor(state, action).items():
                    coefficients[indexDict[state]][indexDict[successor]] -= (probability * self.discount_factor)
                    RightHandSide[indexDict[state]] += (probability * self.mdp.get_reward(state, action, successor))
        x = np.linalg.lstsq(coefficients, RightHandSide, rcond=None)[0]
        for state in self.utilities.keys():
            self.utilities[state] = x[indexDict[state]]

    # Applies a single utility update followed by a single policy update
    # then returns True if the policy has converged and False otherwise
    def update(self) -> bool:
        """
        after applying both utility update and policy update,
        we check if the policies converges and return False if not
        """
        policies = self.policy.copy()
        self.update_utilities()
        self.update_policy()
        if policies != self.policy:
            return False
        return True

    # This function applies value iteration starting from the current utilities stored in the agent and stores the new utilities in the agent
    # NOTE: this function does incremental update and does not clear the utilities to 0 before running
    # In other words, calling train(M) followed by train(N) is equivalent to just calling train(N+M)
    def train(self, iterations: Optional[int] = None) -> int:
        iteration = 0
        while iterations is None or iteration < iterations:
            iteration += 1
            if self.update():
                break
        return iteration

    # Given an environment and a state, return the best action as guided by the learned utilities and the MDP
    # If the state is terminal, return None
    def act(self, env: Environment[S, A], state: S) -> A:
        if self.mdp.is_terminal(state):
            return None
        return self.policy[state]

    # Save the utilities to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            utilities = {self.mdp.format_state(state): value for state, value in self.utilities.items()}
            policy = {
                self.mdp.format_state(state): (None if action is None else self.mdp.format_action(action))
                for state, action in self.policy.items()
            }
            json.dump({
                "utilities": utilities,
                "policy": policy
            }, f, indent=2, sort_keys=True)

    # loads the utilities from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.utilities = {self.mdp.parse_state(state): value for state, value in data['utilities'].items()}
            self.policy = {
                self.mdp.parse_state(state): (None if action is None else self.mdp.parse_action(action))
                for state, action in data['policy'].items()
            }
