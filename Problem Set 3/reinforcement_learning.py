from typing import Callable, DefaultDict, Dict, Generic, List, Optional, Union
from agents import Agent
from environment import Environment, S, A
from helpers.mt19937 import RandomGenerator

import json
from collections import defaultdict

from helpers.utils import NotImplemented

# The base class for all Reinforcement Learning Agents required for this problem set
class RLAgent(Agent[S, A]):
    rng: RandomGenerator # A random number generator used for exploration
    actions: List[A] # A list of all actions that the environment accepts
    discount_factor: float # The discount factor "gamma"
    epsilon: float # The exploration probability for epsilon-greedy
    learning_rate: float # The learning rate "alpha"

    def __init__(self,
            actions: List[A], 
            discount_factor: float = 0.99, 
            epsilon: float = 0.5, 
            learning_rate: float = 0.01, 
            seed: Optional[int] = None) -> None:
        super().__init__()
        self.rng = RandomGenerator(seed) # initialize the random generator with a seed for reproducability
        self.actions = actions
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    # A virtual function that returns the Q-value for a specific state and action
    # This should be overriden by the derived RL agents
    def compute_q(self, env: Environment[S, A], state: S, action: A) -> float:
        return 0
    
    # Returns true if we should explore (rather than exploit)
    def should_explore(self) -> bool:
        return self.rng.float() < self.epsilon

    def act(self, env: Environment[S, A], observation: S, training: bool = False) -> A:
        actions = env.actions()
        if training and self.should_explore():
            return actions[self.rng.int(0, len(actions) - 1)]
        else:
            q_actions = [(self.compute_q(env, observation, action), action) for action in actions]
            q_actions.sort(key=lambda item: -item[0])
            return q_actions[0][1]

#############################
#######     SARSA      ######
#############################

# This is a class for a generic SARSA agent
class SARSALearningAgent(RLAgent[S, A]):
    Q: DefaultDict[S, DefaultDict[A, float]] # The table of the Q values
                                                 # The first key is the string representation of the state
                                                 # The second key is the string representation of the action
                                                 # The value is the Q-value of the given state and action
    
    def __init__(self, 
            actions: List[A], 
            discount_factor: float = 0.99, 
            epsilon: float = 0.5, 
            learning_rate: float = 0.01, 
            seed: Optional[int] = None) -> None:
        super().__init__(actions, discount_factor, epsilon, learning_rate, seed)
        self.Q = defaultdict(lambda:defaultdict(lambda:0)) # The default Q value is 0

    def compute_q(self, env: Environment[S, A], state: S, action: A) -> float:
        return self.Q[state][action] # Return the Q-value of the given state and action
        # NOTE: we cast the state and the action to a string before querying the dictionaries
    
    # Update the value of Q(state, action) using this transition via the SARSA update rule
    def update(self, env: Environment[S, A], state: S, action: A, reward: float, next_state: S, next_action: Optional[A]):
        """
        :param env:
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param next_action:
        :return: update the Q value of state and action
        """
        """
        we simply apply the equation for calculating the Q values in SARSA algorithm
        """
        lr = self.learning_rate
        Qnext = 0
        gamma = self.discount_factor
        thisQ = self.Q[state][action]
        if next_action is not None:
            Qnext = self.compute_q(env, next_state, next_action)
        self.Q[state][action] = thisQ + lr*(reward + (gamma*Qnext) - thisQ)

    # Save the Q-table to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            Q = {
                env.format_state(state) : {
                    env.format_action(action): value for action, value in state_q.items()
                } for state, state_q in self.Q.items()
            }
            json.dump(Q, f, indent=2, sort_keys=True)
    
    # load the Q-table from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            Q = json.load(f)
            self.Q = {
                env.parse_state(state) : {
                    env.parse_action(action): value for action, value in state_q.items()
                } for state, state_q in Q.items()
            }

#############################
#####   Q-Learning     ######
#############################

# This is a class for a generic Q-learning agent
class QLearningAgent(RLAgent[S, A]):
    Q: DefaultDict[str, DefaultDict[str, float]] # The table of the Q values
                                                 # The first key is the string representation of the state
                                                 # The second key is the string representation of the action
                                                 # The value is the Q-value of the given state and action
    
    def __init__(self, 
            actions: List[A], 
            discount_factor: float = 0.99, 
            epsilon: float = 0.5, 
            learning_rate: float = 0.01, 
            seed: Optional[int] = None) -> None:
        super().__init__(actions, discount_factor, epsilon, learning_rate, seed)
        self.Q = defaultdict(lambda:defaultdict(lambda:0)) # The default Q value is 0

    def compute_q(self, env: Environment[S, A], state: S, action: A) -> float:
        return self.Q[state][action] # Return the Q-value of the given state and action
        # NOTE: we cast the state and the action to a string before querying the dictionaries
    
    # Given a state, compute and return the utility of the state using the function "compute_q"
    def compute_utility(self, env: Environment[S, A], state: S) -> float:
        """
        :param env:
        :param state:
        :return: loop on actions to get maximum Q value
        """
        return max([self.compute_q(env, state, action) for action in env.actions()])

    # Update the value of Q(state, action) using this transition via the Q-Learning update rule
    def update(self, env: Environment[S, A], state: S, action: A, reward: float, next_state: S, done: bool):
        """
        :param env:
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return: update the Q value of state and action
        """
        """
        we simply apply the equation for calculating the Q values in Q learning algorithm using compute_utility 
        to the help in getting the maximum expected future reward 
        """
        lr = self.learning_rate
        gamma = self.discount_factor
        Qnext = 0
        if not done:
            Qnext = self.compute_utility(env, next_state)
        thisQ = self.Q[state][action]

        self.Q[state][action] = thisQ + lr*(reward + (gamma*Qnext) - thisQ)
    
    # Save the Q-table to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            Q = {
                env.format_state(state) : {
                    env.format_action(action): value for action, value in state_q.items()
                } for state, state_q in self.Q.items()
            }
            json.dump(Q, f, indent=2, sort_keys=True)
    
    # load the Q-table from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            Q = json.load(f)
            self.Q = {
                env.parse_state(state) : {
                    env.parse_action(action): value for action, value in state_q.items()
                } for state, state_q in Q.items()
            }

#########################################
#####   Approximate Q-Learning     ######
#########################################

# The type definition for a set of features representing a state
# The key is the feature name and the value is the feature value
Features = Dict[str, float]

# This class takes a state and returns a set of features
class FeatureExtractor(Generic[S, A]):

    # Returns a list of feature names.
    # This will be used by the Approximate Q-Learning agent to initialize its weights dictionary.
    @property
    def feature_names(self) -> List[str]:
        return []
    
    # Given an enviroment and an observation (a state), return a set of features that represent the given state
    def extract_features(self, env: Environment[S, A], state: S) -> Features:
        return {}

# This is a class for a generic Q-learning agent
class ApproximateQLearningAgent(RLAgent[S, A]):
    weights: Dict[A, Features]    # The weights dictionary for this agent.
                                    # The first key is action and the second key is the feature name
                                    # The value is the weight
    feature_extractor: FeatureExtractor[S, A]   # The feature extractor used to extract the features corresponding to a state

    def __init__(self, 
            feature_extractor: FeatureExtractor[S, A],
            actions: List[A], 
            discount_factor: float = 0.99, 
            epsilon: float = 0.5,
            learning_rate: float = 0.01,
            seed: Optional[int] = None) -> None:
        super().__init__(actions, discount_factor, epsilon, learning_rate, seed)
        feature_names = feature_extractor.feature_names
        self.weights = {action:{feature: 0 for feature in feature_names} for action in actions} # we initialize the weights to 0
        self.feature_extractor = feature_extractor

    # Given the features of state and an action, compute and return the Q value
    def __compute_q_from_features(self, features: Dict[str, float], action: A) -> float:
        """
        :param features:
        :param action:
        :return: loop on the features dictionary and return the sum of the weights multiplied by the value of the features
        """
        return sum([self.weights[action][feature] * value for feature, value in features.items()])

    # Given the features of a state, compute and return the utility of the state using the function "__compute_q_from_features"
    def __compute_utility_from_features(self, features: Dict[str, float]) -> float:
        """
        :param features:
        :return: loop on action and return the maximum Q using compute_q_from_feature
        """
        return max([self.__compute_q_from_features(features, action) for action in self.actions])

    def compute_q(self, env: Environment[S, A], state: S, action: A) -> float:
        features = self.feature_extractor.extract_features(env, state)
        return self.__compute_q_from_features(features, action)

    # Update the value of Q(state, action) using this transition via the Q-Learning update rule
    def update(self, env: Environment[S, A], state: S, action: A, reward: float, next_state: S, done: bool):
        """
        :param env:
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return: Update the value of Q(state, action) using this transition via the Q-Learning update rule
        """
        lr = self.learning_rate
        gamma = self.discount_factor
        thisQ = self.compute_q(env, state, action)
        Qnext = 0
        if not done:
           Qnext = self.__compute_utility_from_features(self.feature_extractor.extract_features(env, next_state))
        for feature, value in self.feature_extractor.extract_features(env,state).items():
            self.weights[action][feature] += lr*(reward + (gamma*Qnext) - thisQ)*value

    # Save the weights to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            weights = {env.format_action(action): w for action, w in self.weights.items()}
            json.dump(weights, f, indent=2, sort_keys=True)
    
    # load the weights from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            weights = json.load(f)
            self.weights = {env.parse_action(action): w for action, w in weights.items()}