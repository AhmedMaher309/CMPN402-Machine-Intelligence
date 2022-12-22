from typing import Tuple
from game import HeuristicFunction, Game, S, A
from helpers.utils import NotImplemented

# TODO: Import any modules you want to use
import math


# All search functions take a problem, a state, a heuristic function and the maximum search depth.
# If the maximum search depth is -1, then there should be no depth cutoff (The expansion should not stop before reaching a terminal state) 

# All the search functions should return the expected tree value and the best action to take based on the search results

# This is a simple search function that looks 1-step ahead and returns the action that lead to highest heuristic value.
# This algorithm is bad if the heuristic function is weak. That is why we use minimax search to look ahead for many steps.
def greedy(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    agent = game.get_turn(state)

    terminal, values = game.is_terminal(state)
    if terminal: return values[agent], None

    actions_states = [(action, game.get_successor(state, action)) for action in game.get_actions(state)]
    value, _, action = max(
        (heuristic(game, state, agent), -index, action) for index, (action, state) in enumerate(actions_states))
    return value, action


# Apply Minimax search and return the game tree value and the best action
# Hint: There may be more than one player, and in all the testcases, it is guaranteed that 
# game.get_turn(state) will return 0 (which means it is the turn of the player). All the other players
# (turn > 0) will be enemies. So for any state "s", if the game.get_turn(s) == 0, it should a max node,
# and if it is > 0, it should be a min node. Also remember that game.is_terminal(s), returns the values
# for all the agents. So to get the value for the player (which acts at the max nodes), you need to
# get values[0].
def minimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    """
    :param game:
    :param state:
    :param heuristic:
    :param max_depth:
    :return: the game tree value and the best action
    """
    player = game.get_turn(state)
    isTerminalState, terminateValues = game.is_terminal(state)
    """ check for the base case and return the value with None as an action """
    if isTerminalState:
        return terminateValues[0], None
    if max_depth == 0:
        """ return the values using heuristic if we reached the maximum depth """
        return heuristic(game, state, 0), None
    maximum = -9999999
    minimum = 9999999
    neededAction = None
    """
      if the player is the max player, then we get the max value and the action leading to it
    """
    if player == 0:
        for eachAction in game.get_actions(state):
            child = game.get_successor(state, eachAction)
            returnedValue = minimax(game, child, heuristic, max_depth - 1)
            if returnedValue[0] > maximum:
                maximum = returnedValue[0]
                neededAction = eachAction
        return maximum, neededAction
    else:
        """
         else the player is not the max player, then we get the min value and the action leading to it
        """
        for eachAction in game.get_actions(state):
            child = game.get_successor(state, eachAction)
            returnedValue = minimax(game, child, heuristic, max_depth - 1)
            if returnedValue[0] < minimum:
                minimum = returnedValue[0]
                neededAction = eachAction
        return minimum, neededAction


# Apply Alpha Beta pruning and return the tree value and the best action
# Hint: Read the hint for minimax.
def alphabeta(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    """
    alphabeta is like the minimax algorithm but the difference between this algorithm and the minimax is the alpha
    and beta checks that enable pruning some states
    """
    alpha = -9999999
    beta = 9999999
    value, action = recursiveAlphaBeta(game, state, heuristic, max_depth, alpha, beta)
    return value, action

def recursiveAlphaBeta(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1, alpha=-99999999,
                       beta=99999999) -> Tuple[float, A]:
    """
    :param game:
    :param state:
    :param heuristic:
    :param max_depth:
    :param alpha:
    :param beta:
    :return: the game tree value and the best action
    """
    player = game.get_turn(state)
    isTerminalState, terminateValues = game.is_terminal(state)
    if isTerminalState:  # check for the base case
        return terminateValues[0], None
    if max_depth == 0:
        return heuristic(game, state, 0), None
    maximum = -9999999
    minimum = 9999999
    neededAction = None
    """
      if the player is the max player, then we get the max value and the action leading to it
    """
    if player == 0:
        for eachAction in game.get_actions(state):
            child = game.get_successor(state, eachAction)
            returnedValue = recursiveAlphaBeta(game, child, heuristic, max_depth - 1, alpha, beta)
            if returnedValue[0] > maximum:
                maximum = returnedValue[0]
                neededAction = eachAction
            alpha = max(alpha, maximum)  # the alpha check for pruning
            if alpha >= beta:
                break
        return maximum, neededAction
    else:
        """
          else the player is not the max player, then we get the min value and the action leading to it
        """
        for eachAction in game.get_actions(state):
            child = game.get_successor(state, eachAction)
            returnedValue = recursiveAlphaBeta(game, child, heuristic, max_depth - 1, alpha, beta)
            if returnedValue[0] < minimum:
                minimum = returnedValue[0]
                neededAction = eachAction
            beta = min(beta, minimum)  # the beta check for pruning
            if beta <= alpha:
                break
        return minimum, neededAction


# Apply Alpha Beta pruning with move ordering and return the tree value and the best action
# Hint: Read the hint for minimax.
def alphabeta_with_move_ordering(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) ->Tuple[float, A]:
    """
    this function is the as same as the alpha beta but with one difference,
    that this function checks the values of the states according to the order using the heuristic function
    """
    alpha = -9999999
    beta = 9999999
    value, action = recursiveAlphaBetaWithOrder(game, state, heuristic, max_depth, alpha, beta)
    return value, action

def recursiveAlphaBetaWithOrder(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1, alpha=-99999999, beta=99999999) -> Tuple[float, A]:
    """
    :param game:
    :param state:
    :param heuristic:
    :param max_depth:
    :param alpha:
    :param beta:
    :return: the game tree value and the best action
    """
    player = game.get_turn(state)
    isTerminalState, terminateValues = game.is_terminal(state)
    if isTerminalState:  # check for the base case
        return terminateValues[0], None
    if max_depth == 0:
        return heuristic(game, state, 0), None
    childrenAndActions = []  # get the children with each the leading actions to each of them
    for action in game.get_actions(state):
        h = heuristic(game, game.get_successor(state, action), 0)
        child = game.get_successor(state, action)
        childrenAndActions.append((child, action, h))
    maximum = -9999999
    minimum = 9999999
    neededAction = None
    """
      if the player is the max player, then we get the max value and the action leading to it
    """
    if player == 0:
        childrenAndActions.sort(key=lambda m: m[2], reverse=True)
        for eachState, eachAction, eachHeuristic in childrenAndActions:
            returnedValue = recursiveAlphaBetaWithOrder(game, eachState, heuristic, max_depth - 1, alpha, beta)
            if returnedValue[0] > maximum:
                maximum = returnedValue[0]
                neededAction = eachAction
            alpha = max(alpha, maximum)  # the alpha check for pruning
            if alpha >= beta:
                break
        return maximum, neededAction
    else:
        """
          else the player is not the max player, then we get the min value and the action leading to it
        """
        childrenAndActions.sort(key=lambda m: m[2])
        for eachState, eachAction, eachHeuristic in childrenAndActions:
            returnedValue = recursiveAlphaBetaWithOrder(game, eachState, heuristic, max_depth - 1, alpha, beta)
            if returnedValue[0] < minimum:
                minimum = returnedValue[0]
                neededAction = eachAction
            beta = min(beta, minimum)  # the beta check for pruning
            if beta <= alpha:
                break
        return minimum, neededAction


# Apply Expectimax search and return the tree value and the best action
# Hint: Read the hint for minimax, but note that the monsters (turn > 0) do not act as min nodes anymore,
# they now act as chance nodes (they act randomly).
def expectimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    """
    :param game:
    :param state:
    :param heuristic:
    :param max_depth:
    :return:the game tree value and the best action
    """

    """
    the function is like the normal minimax, but it deals with the probability states
     by calculating the weighted average
    """
    player = game.get_turn(state)
    isTerminalState, terminateValues = game.is_terminal(state)
    if isTerminalState:  # check for the base case
        return terminateValues[0], None
    if max_depth == 0:
        return heuristic(game, state, 0), None
    maximum = -999999999
    minimum = 0
    neededAction = None
    if player == 0:
        for action in game.get_actions(state):
            child = game.get_successor(state, action)
            returnedValue = expectimax(game, child, heuristic, max_depth - 1)
            if returnedValue[0] > maximum:
                maximum = returnedValue[0]
                neededAction = action
        return maximum, neededAction
    else:
        for action in game.get_actions(state):
            child = game.get_successor(state, action)
            returnedValue = expectimax(game, child, heuristic, max_depth - 1)
            minimum = returnedValue[0] + minimum
        return minimum / (len(game.get_actions(state))), None  # return the weighted average , with the action
