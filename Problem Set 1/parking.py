from typing import Any, Dict, Set, Tuple, List
from problem import Problem
from mathutils import Direction, Point
from helpers import utils

# TODO: (Optional) Instead of Any, you can define a type for the parking state
ParkingState = Tuple[Point]
# An action of the parking problem is a tuple containing an index 'i' and a direction 'd' where car 'i' should move in the direction 'd'.
ParkingAction = Tuple[int, Direction]


# This is the implementation of the parking problem
class ParkingProblem(Problem[ParkingState, ParkingAction]):
    passages: Set[
        Point]  # A set of points which indicate where a car can be (in other words, every position except walls).
    cars: Tuple[Point]  # A tuple of points where state[i] is the position of car 'i'.
    slots: Dict[
        Point, int]  # A dictionary which indicate the index of the parking slot (if it is 'i' then it is the lot of car 'i') for every position.
    # if a position does not contain a parking slot, it will not be in this dictionary.
    width: int  # The width of the parking lot.
    height: int  # The height of the parking lot.

    # This function should return the initial state
    def get_initial_state(self) -> ParkingState:
        return self.cars

    # This function should return True if the given state is a goal. Otherwise, it should return False.
    def is_goal(self, state: ParkingState) -> bool:
        """
        :param state:
        :return: True if we reached the goal state
        """
        for point in state:  # check if each point of the current state is in the slots of cars, if no --> return false
            if point in self.slots.keys():
                pass
            else:
                return False
        for i in range(len(state)):  # now check if each car in its desired slot , if yes --> this the goal state
                                     # if no --> we did not reach goal state yet
            if state[i] in self.slots.keys():
                if i == self.slots[state[i]]:
                    pass
                else:
                    return False
        return True

    # This function returns a list of all the possible actions that can be applied to the given state
    def get_actions(self, state: ParkingState) -> List[ParkingAction]:
        """
        we get all the possible actions for all cars and push them to a list then return this list
        :param state: state of the cars now
        :return: list of all the possible actions that can be applied to the given state
        """
        actions_list = []
        for i in range(len(state)):
            if (state[i] + Point(1, 0)) in self.passages and (
                    state[i] + Point(1, 0)) not in state:  # check it is able to move right
                park_action = (i, 'R')
                actions_list.append(park_action)
            if (state[i] + Point(-1, 0)) in self.passages and (
                    state[i] + Point(-1, 0)) not in state:  # check it is able to move left
                park_action = (i, 'L')
                actions_list.append(park_action)
            if (state[i] + Point(0, 1)) in self.passages and (
                    state[i] + Point(0, 1)) not in state:  # check it is able to move up
                park_action = (i, 'D')
                actions_list.append(park_action)
            if (state[i] + Point(0, -1)) in self.passages and (
                    state[i] + Point(0, -1)) not in state:  # check it is able to move down
                park_action = (i, 'U')
                actions_list.append(park_action)
        return actions_list

    # This function returns a new state which is the result of applying the given action to the given state
    def get_successor(self, state: ParkingState, action: ParkingAction) -> ParkingState:
        """
        :param state: is the current state of the parking
        :param action: is the change in the state
        :return: the state after changing it by the action
        """
        state_list = list(state)  # convert the tuple to list to make it mutable
        state_list[action[0]] += action[1].to_vector()   # add the action
        state = tuple(state_list)   # reconvert to tuple again
        return state

    # This function returns the cost of applying the given action to the given state
    def get_cost(self, state: ParkingState, action: ParkingAction) -> float:
        """
        check if the state after modifying it by the action will be a slot of another car 
        which will cause additional cost to the ordinary cost (1) equal to 100 
        then return the cost 
        """
        cost = 1
        if state[action[0]] + action[1].to_vector() in self.slots.keys() and \
                self.slots[state[action[0]] + action[1].to_vector()] != action[0]:
            cost = 101
        return cost

    # Read a parking problem from text containing a grid of tiles
    @staticmethod
    def from_text(text: str) -> 'ParkingProblem':
        passages = set()
        cars, slots = {}, {}
        lines = [line for line in (line.strip() for line in text.splitlines()) if line]
        width, height = max(len(line) for line in lines), len(lines)
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char != "#":
                    passages.add(Point(x, y))
                    if char == '.':
                        pass
                    elif char in "ABCDEFGHIJ":
                        cars[ord(char) - ord('A')] = Point(x, y)
                    elif char in "0123456789":
                        slots[int(char)] = Point(x, y)
        problem = ParkingProblem()
        problem.passages = passages
        problem.cars = tuple(cars[i] for i in range(len(cars)))
        problem.slots = {position: index for index, position in slots.items()}
        problem.width = width
        problem.height = height
        return problem

    # Read a parking problem from file containing a grid of tiles
    @staticmethod
    def from_file(path: str) -> 'ParkingProblem':
        with open(path, 'r') as f:
            return ParkingProblem.from_text(f.read())
