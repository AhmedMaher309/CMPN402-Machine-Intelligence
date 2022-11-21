import queue

from problem import HeuristicFunction, Problem, S, A, Solution
from collections import deque
from helpers import utils
from queue import PriorityQueue


# TODO: Import any modules you want to use

# All search functions take a problem and a state
# If it is an informed search function, it will also receive a heuristic function
# S and A are used for generic typing where S represents the state type and A represents the action type

# All the search functions should return one of two possible type:
# 1. A list of actions which represent the path from the initial state to the final state
# 2. None if there is no solution


def BreadthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    """
    get the path from a source node (initial_state)
    to the goal node in the graph (problem) using the BFS algorithm
    :param problem: the graph we traverse through
    :param initial_state: the source node we start from
    :return: the path to the goal in a form of list
    """
    queue = []  # list of frontiers
    visited = []  # list of the visited nodes of the graph
    my_parent = {initial_state: "none"}
    sol = []  # solution path that will be returned
    queue.append(initial_state)
    visited.append(initial_state)
    while queue:
        source = queue.pop(0)
        if problem.is_goal(source):
            sol.append(source)
            break  # break when reach the goal state
        for adjacent in problem.get_actions(source):
            next_state = problem.get_successor(source, adjacent)
            if next_state not in visited:  # check if the state is not visited (explored) then we prepare it for the expansion
                my_parent[next_state] = source
                visited.append(next_state)
                queue.append(next_state)
    """
    searching for the goal node and iterating backward on the dictionary 
    to get its ancestors
    """
    final = []
    if len(sol) != 0:
        father = my_parent[sol[0]]
        while father != "none":
            sol.append(father)
            father = my_parent[father]
        new = sol  # the list of states from source to goal
        new.reverse()
        for i in range(len(new)-1):   # extract each action the lead every state to go to the next state after it in the list
                                      # and return this list of actions
            for action in problem.get_actions(new[i]):
                if problem.get_successor(new[i], action) == new[i+1]:
                    final.append(action)
        return final
    else:
        return None

"""/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"""

def DepthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    """
    get the path from a source node (initial_state)
    to the goal node in the graph (problem) using the DFS algorithm
    :param problem: the graph we traverse through
    :param initial_state: the source node we start from
    :return: the path to the goal in a form of list
    """
    visited = set()
    my_parent = {initial_state: "none"}
    sol = []
    dfs(problem, initial_state, visited, my_parent)  # call the utility function to apply DFS recursively
    for nodes in my_parent.keys():
        if problem.is_goal(nodes):
            sol.append(nodes)
            break                        # break when reach the goal state
    final = []
    if len(sol) != 0:
        father = my_parent[sol[0]]
        while father != "none":
            sol.append(father)
            father = my_parent[father]
        new = sol                       # the list of states from source to goal
        new.reverse()
        for i in range(len(new) - 1):   # extract each action the lead every state to go to the next state after it in the list
                                        # and return this list of actions
            for action in problem.get_actions(new[i]):
                if problem.get_successor(new[i], action) == new[i + 1]:
                    final.append(action)
        return final
    else:
        return None


"""
a utility function that do the dfs by recursion and then receive the results
on the visited list and my_parent dictionary
"""
def dfs(problem, source, visited, parent_dict):  # apply recursive DFS to get the path of the goal state
    visited.add(source)
    for adjacent in problem.get_actions(source):
        next_state = problem.get_successor(source, adjacent)
        if next_state not in visited:
            parent_dict[next_state] = source
            dfs(problem, next_state, visited, parent_dict)


"""/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"""

def UniformCostSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    """
        get the best path from a source node (initial_state)
        to the goal node in the graph (problem) using the UCS algorithm
        :param problem: the graph we traverse through
        :param initial_state: the source node we start from
        :return: the path to the goal in a form of list
        """
    sol = []
    qu = queue.PriorityQueue()  # queue that holds each node, the order of it in the queue, and the cost of the path to it
    count = 0
    """
    we use count as the second element in the priority as if we got two nodes with same cost 
    then the queue compare using the insertion order.
    """
    qu.put((0, count, initial_state))
    visited = [initial_state]  # list of the visited nodes (finished expansion)
    being_processed = []  # list of the nodes that was expanded but may be expanded again( have a lower path cost)
    my_parent = {initial_state: "none"}
    while len(qu.queue) > 0:
        source = qu.get()
        if source in being_processed:  # remove the state from the being_processed list and put in visited,
                                       # means we don't want to revisit it again
            being_processed.remove(source)
            visited.append(source)
        if problem.is_goal(source[2]):
            sol.append(source[2])
            break  # break the loop when reach the goal
        for adjacent in problem.get_actions(source[2]):
            """
            case 1: the node is not in visited or in being_processed, so we add it to the being_processed list
            in order to prepare it for expansion 

            case 2: the node is in the being_processed list but not in the visited list, so we check if the current path cost
            to this node is bigger than its previous pass cost, if it is true we modify it.
            """
            next_state = problem.get_successor(source[2], adjacent)
            if next_state not in visited and next_state not in being_processed:  # beginning of first case check
                my_parent[next_state] = source[2]
                being_processed.append(next_state)
                cost = problem.get_cost(source[2], next_state) + source[0]
                count += 1
                qu.put((cost, count, next_state))
            elif next_state in being_processed:   # beginning of second case check
                cost = problem.get_cost(source[2], next_state) + source[0]
                my_cost = check_cost(qu, next_state)
                if my_cost > cost:
                    qu.queue.remove((my_cost, count, next_state))  # remove the node with its old cost
                    count += 1
                    qu.put((cost, count, next_state))   # add the node with its new cost
                    my_parent[next_state] = source[2]
    final = []
    if len(sol) != 0:
        father = my_parent[sol[0]]
        while father != "none":
            sol.append(father)
            father = my_parent[father]
        new = sol  # the list of states from source to goal
        new.reverse()
        for i in range(len(new) - 1):   # extract each action the lead every state to go to the next state after it in the list
                                        # and return this list of actions
            for action in problem.get_actions(new[i]):
                if problem.get_successor(new[i], action) == new[i + 1]:
                    final.append(action)
        return final
    else:
        return None

"""/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"""

"""
utility function for the uniform cost search 
the function search for a specific node in the queue return its cost
"""
def check_cost(que: PriorityQueue(), elt):
    que_list = list(que.queue)
    for node in que_list:
        if node[2] == elt:
            return node[0]
    return 0

"""/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"""
def AStarSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    """
    get the best path from a source node (initial_state)
    to the goal node in the graph (problem) using the A* search algorithm
    :param heuristic:
    :param problem: the graph we traverse through
    :param initial_state: the source node we start from
    :return: the path to the goal in a form of list
    """
    sol = []
    qu = queue.PriorityQueue()  # queue that holds each node, the order of it in the queue, and the cost of the path to it
    """
    we use count as the second element in the priority as if we got two nodes with same cost 
    then the queue compare using the insertion order.
    """
    count = 0
    qu.put((heuristic(problem, initial_state), count, initial_state))
    visited = set()  # list of the visited nodes (finished expansion)
    visited.add(initial_state)
    being_processed = []  # list of the nodes that was expanded but may be expanded again( have a lower path cost)
    my_parent = {initial_state: "none"}
    while len(qu.queue) > 0:
        source = qu.get()
        if source in being_processed:    # remove the state from the being_processed list and put in visited,
                                         # means we don't want to revisit it again
            being_processed.remove(source)
            visited.add(source)
        if problem.is_goal(source[2]):
            sol.append(source[2])
            break  # break the loop when reach the goal
        for adjacent in problem.get_actions(source[2]):
            """
            case 1: the node is not in visited or in being_processed, so we add it to the being_processed list
            in order to prepare it for expansion 
            
            case 2: the node is in the being_processed list but not in the visited list, so we check if the current path cost
            to this node is bigger than its previous pass cost, if it is true we modify it.
            """
            next_state = problem.get_successor(source[2], adjacent)
            if next_state not in visited and next_state not in being_processed:  # beginning of first case check
                my_parent[next_state] = source[2]
                being_processed.append(next_state)
                cost = problem.get_cost(source[2], next_state) + source[0] + heuristic(problem, next_state) - heuristic(problem, source[2])
                count += 1
                qu.put((cost, count, next_state))
            elif next_state in being_processed:   # beginning of second case check
                cost = problem.get_cost(source[2], next_state) + source[0] + heuristic(problem, next_state) - heuristic(problem, source[2])
                my_cost = check_cost(qu, next_state)  # use the utility function to get the cost of the node
                if my_cost > cost:  # compare the costs to take decision based on the result
                    if (my_cost, count, next_state) in qu.queue:
                        qu.queue.remove((my_cost, count, next_state))  # remove the node with its old cost
                        count += 1
                        qu.put((cost, count, next_state))   # add the node with its new cost
                        my_parent[next_state] = source[2]
    final = []
    if len(sol) != 0:
        father = my_parent[sol[0]]
        while father != "none":
            sol.append(father)
            father = my_parent[father]
        new = sol  # the list of states from source to goal
        new.reverse()
        for i in range(len(new) - 1):    # extract each action the lead every state to go to the next state after it in the list
                                         # and return this list of actions
            for action in problem.get_actions(new[i]):
                if problem.get_successor(new[i], action) == new[i + 1]:
                    final.append(action)
        return final
    else:
        return None

"""/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"""

def BestFirstSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    """
    get the best path from a source node (initial_state)
    to the goal node in the graph (problem) using the Best First Sesrch algorithm
    :param heuristic:
    :param problem: the graph we traverse through
    :param initial_state: the source node we start from
    :return: the path to the goal in a form of list
    """
    sol = []
    qu = queue.PriorityQueue()  # queue that holds each node, the order of it in the queue, and the cost of the path to it
    """
    The count is useless in this algorithm
    but we kept it to avoid many modifications as the function uses the utility check_cost
    """
    count = 0
    qu.put((heuristic(problem, initial_state), count, initial_state))
    visited = set()  # list of the visited nodes (finished expansion)
    visited.add(initial_state)
    my_parent = {initial_state: "none"}
    while len(qu.queue) > 0:
        source = qu.get()
        if problem.is_goal(source[2]):
            sol.append(source[2])
            break  # break the loop when reach the goal
        for adjacent in problem.get_actions(source[2]):
            """
            if the node is not in visited, we add it to the visited list
            in order to prepare it for expansion 
            """
            next_state = problem.get_successor(source[2], adjacent)
            if next_state not in visited:
                my_parent[next_state] = source[2]
                visited.add(next_state)
                cost = heuristic(problem, next_state)
                count += 1
                qu.put((cost, count, next_state))
    final = []
    if len(sol) != 0:
        father = my_parent[sol[0]]
        while father != "none":
            sol.append(father)
            father = my_parent[father]
        new = sol  # the list of states from source to goal
        new.reverse()
        for i in range(len(new) - 1):  # extract each action the lead every state to go to the next state after it in the list
                                       # and return this list of actions
            for action in problem.get_actions(new[i]):
                if problem.get_successor(new[i], action) == new[i + 1]:
                    final.append(action)
        return final
    else:
        return None
