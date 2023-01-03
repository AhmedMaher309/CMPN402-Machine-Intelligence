from typing import Any, Dict, List, Optional
from CSP import Assignment, BinaryConstraint, Problem, UnaryConstraint
from helpers.utils import NotImplemented
import copy


# This function should apply 1-Consistency to the problem.
# In other words, it should modify the domains to only include values that satisfy their variables' unary constraints.
# Then all unary constraints should be removed from the problem (they are no longer needed).
# The function should return False if any domain becomes empty. Otherwise, it should return True.
def one_consistency(problem: Problem) -> bool:
    """
    :param problem:
    :return: True if the applying unary constraints won't make the problem unsolvable
    """

    """
    1- loop on all the variables
    2- loop on all the constraints
    3- check if the constraint is unary constraint and it is for that variable
    4- get all the values of this variable domain,
    and delete only the values that will violate this unary constraint
    5- last step is to delete all the unary constraints.
    """
    for variable in problem.variables:
        for constrain in problem.constraints:
            if isinstance(constrain, UnaryConstraint) and constrain.variable == variable:
                values = list(problem.domains[variable])
                for value in values:
                    valueSingleDict = {variable: value}
                    if constrain.is_satisfied(valueSingleDict):
                        pass
                    else:
                        values.remove(value)
                    if len(values) == 0:
                        return False
                problem.domains[variable] = set(values)
    for constrain in problem.constraints.copy():
        if isinstance(constrain, UnaryConstraint):
            problem.constraints.remove(constrain)
    return True


# This function should implement forward checking
# The function is given the problem, the variable that has been assigned and its assigned value and the domains of the unassigned values
# The function should return False if it is impossible to solve the problem after the given assignment, and True otherwise.
# In general, the function should do the following:
#   - For each binary constraints that involve the assigned variable:
#       - Get the other involved variable.
#       - If the other variable has no domain (in other words, it is already assigned), skip this constraint.
#       - Update the other variable's domain to only include the values that satisfy the binary constraint with the assigned variable.
#   - If any variable's domain becomes empty, return False. Otherwise, return True.
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
def forward_checking(problem: Problem, assigned_variable: str, assigned_value: Any, domains: Dict[str, set]) -> bool:
    """
    :param problem:
    :param assigned_variable:
    :param assigned_value:
    :param domains:
    :return: True, only if applying forward checking won't clear a variable domain,
     which means that the problem is not solvable
    """

    """
    1- loop on all the constraints
    2- check if the constraint is for that assigned_variable
    3- get the other variable that share the constraint with assigned_variable.
    4- check if assigning the assigned_value to the assigned_variable will violate the binary constraint,
     if yes delete this value from the domain of the other variable
    """
    for constraint in problem.constraints:
        if assigned_variable in constraint.variables:
            otherVariable = constraint.get_other(assigned_variable)
            if otherVariable in domains.keys():
                for value in domains[otherVariable].copy():
                    if not constraint.is_satisfied({assigned_variable: assigned_value, otherVariable: value}):
                        domains[otherVariable].remove(value)
                        if len(domains[otherVariable]) == 0:
                            return False
    return True


# This function should return the domain of the given variable order based on the "least restraining value" heuristic.
# IMPORTANT: This function should not modify any of the given arguments.
# Generally, this function is very similar to the forward checking function, but it differs as follows:
#   - You are not given a value for the given variable, since you should do the process for every value in the variable's
#     domain to see how much it will restrain the neigbors domain
#   - Here, you do not modify the given domains. But you can create and modify a copy.
# IMPORTANT: If multiple values have the same priority given the "least restraining value" heuristic, 
#            order them in ascending order (from the lowest to the highest value).
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
def least_restraining_values(problem: Problem, variable_to_assign: str, domains: Dict[str, set]) -> List[Any]:
    """
    :param problem:
    :param variable_to_assign:
    :param domains:
    :return: list of the domain values of variable_to_assign sorted according to the least restraining value
    """

    """
    1- loop on the domains of each variable 
    2- check on each domain if it has any constraints with the values of variable_to_assign 
    3- if yes, update the value of the variable_to_assign frequency to 
    4- sort the domain of the variable_to_assign according to the least restraining values
    5- return that sorted domain 
    """
    domainFrequency = {}
    for value in domains[variable_to_assign]:
        domainFrequency[value] = 0
    for value in domainFrequency.keys():
        for constraint in problem.constraints:
            if variable_to_assign in constraint.variables:
                otherVariable = constraint.get_other(variable_to_assign)
                if otherVariable in domains.keys():            # check if this variable is in the domains
                    for eachValue in domains[otherVariable]:
                        if not constraint.is_satisfied({variable_to_assign: value, otherVariable: eachValue}):
                            domainFrequency[value] += 1
    sort_orders = sorted(domainFrequency.items(), key=lambda x: x[1], reverse=False)
    sortedValues = []
    for ans in sort_orders:
        sortedValues.append(ans[0])
    return sortedValues


# This function should return the variable that should be picked based on the MRV heuristic.
# IMPORTANT: This function should not modify any of the given arguments.
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
# IMPORTANT: If multiple variables have the same priority given the MRV heuristic, 
#            order them in the same order in which they appear in "problem.variables".
def minimum_remaining_values(problem: Problem, domains: Dict[str, set]) -> str:
    """
    :param problem:
    :param domains:
    :return: the variable with the minimum remaining values
    """

    """
    return the variable with the smallest domain length 
    """
    minimum = 999999999
    returnedVariable = None
    for variable in domains.keys():
        if len(domains[variable]) < minimum:
            minimum = len(domains[variable])
            returnedVariable = variable
    return returnedVariable


# This function should solve CSP problems using backtracking search with forward checking.
# The variable ordering should be decided by the MRV heuristic.
# The value ordering should be decided by the "least restraining value" heurisitc.
# Unary constraints should be handled using 1-Consistency before starting the backtracking search.
# This function should return the first solution it finds (a complete assignment that satisfies the problem constraints).
# If no solution was found, it should return None.
# IMPORTANT: To get the correct result for the explored nodes, you should check if the assignment is complete only once using "problem.is_complete"
#            for every assignment including the initial empty assignment, EXCEPT for the assignments pruned by the forward checking.
#            Also, if 1-Consistency deems the whole problem unsolvable, you shouldn't call "problem.is_complete" at all.
def solve(problem: Problem) -> Optional[Assignment]:
    """
    :param problem:
    :return:the first solution the function finds, which is the first successful assign
    """
    if not one_consistency(problem):
        return None
    assign = {}
    if RecursiveTrials(problem, assign, problem.domains):
        return assign
    else:
        return None

def RecursiveTrials(problem, assign: Assignment, domains):
    """
    :param problem:
    :param assign:
    :param domains:
    :return: true or false according to if the backtracking was successful at any assign,
    and store this assign in one of its parameters
    """

    """
    A utility function for the solve problem that do the actual backtracking and return the true or false to the solve,
    with modifying its parameters
    """
    if problem.is_complete(assign):
        return True
    variableChosen = minimum_remaining_values(problem, domains)
    values = least_restraining_values(problem, variableChosen, domains)
    domainCopy = copy.deepcopy(domains)
    del domains[variableChosen]
    for value in values:
        assign[variableChosen] = value
        if forward_checking(problem, variableChosen, value, domains):
            if RecursiveTrials(problem, assign, domains):
                return True
            else:
                domains = copy.deepcopy(domainCopy)
                if variableChosen in domains.keys():
                    del domains[variableChosen]
    return False



