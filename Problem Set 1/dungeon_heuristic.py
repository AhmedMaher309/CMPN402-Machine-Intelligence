from dungeon import DungeonProblem, DungeonState
from mathutils import Direction, Point, euclidean_distance, manhattan_distance
from helpers import utils
from search import BreadthFirstSearch


# This heuristic returns the distance between the player and the exit as an estimate for the path cost
# While it is consistent, it does a bad job at estimating the actual cost thus the search will explore a lot of nodes before finding a goal
def weak_heuristic(problem: DungeonProblem, state: DungeonState):
    return euclidean_distance(state.player, problem.layout.exit)

# First trial
def strong_heuristic(problem: DungeonProblem, state: DungeonState) -> float:
    direct_distance = manhattan_distance(state.player, problem.layout.exit)
    """
    The idea is to get an intermediate point between the start and end goal
    so we can calculate more realistic distance to the goal
    """
    dist_1 = []
    dist_2 = []
    dist_final = []
    if len(state.remaining_coins) >= 1:
        for coin in state.remaining_coins:
            coin_distance = euclidean_distance(state.player, coin)
            dist_1.append(coin_distance)

        for coin in state.remaining_coins:
            coin_end = manhattan_distance(coin, problem.layout.exit)
            dist_2.append(coin_end)
        minimum = 0
        for i in range(len(dist_1)):
            i_dist = dist_1[i] + dist_2[i]
            dist_final.append(i_dist)
            if i_dist > minimum:
                minimum = i_dist
        return minimum

    return direct_distance

# Second trial
def strong_heuristic(problem: DungeonProblem, state: DungeonState) -> float:
    mid_x = problem.layout.exit.x / 2
    mid_y = problem.layout.exit.y / 2
    mid = Point(mid_x, mid_y)
    mid_1 = Point(mid_x + 1, mid_y + 1)
    direct_distance = manhattan_distance(state.player, problem.layout.exit)
    if len(state.remaining_coins) >= 1:
        intermediate_distance = 0
        for coin in state.remaining_coins:
            coin_distance = manhattan_distance(state.player, coin) + \
                            manhattan_distance(coin, mid) +\
                            manhattan_distance(mid_1, problem.layout.exit)
            if coin_distance > intermediate_distance:
                intermediate_distance = coin_distance
        return intermediate_distance

    return direct_distance

# Third trial
def strong_heuristic(problem: DungeonProblem, state: DungeonState) -> float:
    """
    The idea is to get an intermediate point between the start and end goal
    so we can calculate more realistic distance to the goal
    """
    direct_distance = manhattan_distance(state.player, problem.layout.exit)
    if len(state.remaining_coins) >= 1:
        intermediate_distance = 0
        for coin in state.remaining_coins:
            coin_distance = manhattan_distance(state.player, coin) + manhattan_distance(coin, problem.layout.exit)
            if coin_distance > intermediate_distance:
                intermediate_distance = coin_distance
        return intermediate_distance

    return direct_distance





