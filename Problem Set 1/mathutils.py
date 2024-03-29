from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator
import math


# the class Point will hold a 2D coordinate on a discrete grid
# We use dataclass with frozen=True to automatically implement:
#   the constructor, the == operator, the hash function and to make the class immutable
# Now it can be added to sets and used as keys in dictionaries
@dataclass(frozen=True)
class Point:
    __slots__ = ('x', 'y')
    x: int
    y: int

    # The following functions implement the operators +, -, negative and str
    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __neg__(self) -> 'Point':
        return Point(-self.x, -self.y)

    def __str__(self) -> str:
        return f'({self.x}, {self.y})'

    # this allow points to be used as iterators such as writing:
    # x, y = point
    # to unpack the Point class into its x and y components
    def __iter__(self) -> Iterator[int]:
        return iter((self.x, self.y))


# This is a helper function to compute the manhattan distance between 2 points
def manhattan_distance(p1: Point, p2: Point) -> int:
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)


# This is a helper function to compute the euclidean distance between 2 points
def euclidean_distance(p1: Point, p2: Point) -> int:
    difference = p1 - p2
    return math.sqrt(difference.x * difference.x + difference.y * difference.y)


# This enum represent 4 directions (RIGHT, UP, LEFT, RIGHT)
class Direction(IntEnum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

    def rotate(self, amount: int = 1) -> 'Direction':
        return Direction((self + amount) % 4)

    # This function converts a direction to a normalized vector    
    def to_vector(self) -> Point:
        return Direction._Vectors[self]

    # Print Direction as a letter 'R', 'U', 'L', or 'D'
    def __str__(self) -> str:
        return 'RULD'[self]

    # Cast a string into a direction
    @classmethod
    def _missing_(cls, value: str):
        return cls({
                       'r': Direction.RIGHT,
                       'u': Direction.UP,
                       'l': Direction.LEFT,
                       'd': Direction.DOWN,
                   }[value.lower()])


# A list where each entry contains the vector pointing in the corresponding direction
Direction._Vectors = [
    Point(1, 0),
    Point(0, -1),
    Point(-1, 0),
    Point(0, 1)
]
