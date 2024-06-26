from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Union, Optional, NewType, Callable
from collections.abc import Iterable, Iterator
from enum import Enum
import math
from .. import olr

T = TypeVar('T')

"""
Interface for all classes that supply data

T - type of suppliable data 
"""


class DataGenerator(ABC, Generic[T], Iterator[T]):
    """
    An abstract class defining the interface for data generators.
    """
    @abstractmethod
    def next(self) -> T:
        """
        Abstract method to retrieve the next data element.
        """
        pass

    def __iter__(self):
        return self

    def __next__(self) -> T:
        """
        Default implementation of magic method supplies data infinitely. It is your
        responsibility to interrupt (break) loop in order to prevent infinite iterations.
        """
        return self.next()


class OptionallyFiniteDataGenerator(DataGenerator[T]):
    """
    A data generator that allows controlling the number of iterations.
    """

    def __init__(self):
        self.__iter = 0
        self.__max_iters = None

        pass

    def set_max_iters(self, max_iters: Optional[int]):
        """
        Set the maximum number of iterations.
        """
        self.__max_iters = max_iters

    def get_max_iters(self) -> Optional[int]:
        """
        Get the maximum number of iterations.
        """
        return self.__max_iters

    def is_iterations_restricted(self) -> bool:
        """
        Checks if iterations are restricted.
        """
        return self.__max_iters is not None

    def reset_iters(self):
        """
        Resets the iteration count.
        """
        self.__iter = 0

    def __next__(self) -> T:
        if self.is_iterations_restricted():
            if self.__iter >= self.get_max_iters():
                raise StopIteration

        val = super().__next__()
        self.__iter += 1

        return val


class ConstDataGenerator(OptionallyFiniteDataGenerator[T]):
    """
    A data generator that yields a constant value.
    """

    def __init__(self, const: T):
        super().__init__()
        self.const = const

    def next(self) -> T:
        return self.const


class IterableDataGenerator(OptionallyFiniteDataGenerator[T]):
    """
    A data generator that yields elements from an iterable.
    """

    def __init__(self, iterable: Iterable[T]):
        super().__init__()
        self.iterator = iter(iterable)

    def next(self) -> T:
        return next(self.iterator)


class GridRange(ABC, Generic[T]):
    """
    Abstract base class for defining a range of values.
    """

    @abstractmethod
    def get_value(self) -> T:
        """
        Get the current value of the range.
        """
        pass

    @abstractmethod
    def within(self) -> bool:
        """
        Checks if the current value is within the range.
        """
        pass

    @abstractmethod
    def step(self) -> T:
        """
        Move to the next value in the range.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the range to its initial state.
        """
        pass


Numeric = NewType('Numeric', Union[int, float])

RangeDirection = Enum('RangeDirection', ['INCREMENTAL', 'DECREMENTAL'])


class NumericGridRange(GridRange[Numeric]):
    """
    Represents a range of numeric values.
    """

    def __init__(self, a: Numeric, b: Numeric, step: Numeric, round_up_to: int = 9):
        if step <= 0:
            raise ValueError("Step must be a positive number")

        self.__a = a
        self.__b = b
        self.__step = step
        self.__round_up_to = round_up_to
        self.__value = self.get_initial_value()

        if a <= b:
            self.__direction = RangeDirection.INCREMENTAL
        else:
            self.__direction = RangeDirection.DECREMENTAL

    @staticmethod
    def unit(a: Numeric, b: Numeric):
        return NumericGridRange(a, b, 1)

    @staticmethod
    def const(const: Numeric):
        return NumericGridRange.unit(const, const)

    def get_value(self) -> Numeric:
        return self.__value

    def get_initial_value(self) -> Numeric:
        return self.__a

    def within(self) -> bool:
        if self.__direction == RangeDirection.INCREMENTAL:
            return self.__value >= self.__a and self.__value < self.__b
        else:
            return self.__value > self.__b and self.__value <= self.__a

    def step(self) -> Numeric:
        if self.__direction == RangeDirection.INCREMENTAL:
            self.__value += self.__step

            if self.__value > self.__b or math.isclose(self.__value, self.__b):
                self.__value = self.__b
        else:
            self.__value -= self.__step

            if self.__value < self.__b or math.isclose(self.__value, self.__b):
                self.__value = self.__b

        self.__value = round(self.__value, self.__round_up_to)

        return self.__value

    def reset(self):
        self.__value = self.get_initial_value()

    def get_direction(self) -> RangeDirection:
        return self.__direction

    def split(self, n: int):
        if self.__a == self.__b:
            return [NumericGridRange(self.__a, self.__b, self.__step)]

        interval = [min(self.__a, self.__b), max(self.__a, self.__b)]

        intervals = self.__split_interval(interval, n)

        return [NumericGridRange(interval[0], interval[1], self.__step) for interval in intervals]

    def __split_interval(self, l, n: int):
        w = (l[1] - l[0]) // n

        return [[l[0] + i * w, l[0] + (i + 1) * w] for i in range(n)]

    def __str__(self) -> str:
        return f'NumericGridRange({self.__a, self.__b, self.__step})'


DimensionRange = NewType('DimensionRange', GridRange[T])
DimensionRanges = NewType('DimensionRanges', list[GridRange[T]])


class EnumerationDirection(ABC):
    """
    Abstract base class for defining enumeration directions.
    """

    @abstractmethod
    def direct(self, ranges: DimensionRanges):
        """
        Direct the enumeration over a list of ranges.
        """
        pass


class ForwardEnumerationDirection(EnumerationDirection):
    """
    Directs enumeration forward through the ranges.
    """

    def direct(self, ranges: DimensionRanges):
        return ranges


class ReversedEnumerationDirection(EnumerationDirection):
    """
    Directs enumeration backward through the ranges.
    """

    def direct(self, ranges: DimensionRanges):
        return reversed(ranges)


class GridIterator(GridRange):
    """
    Iterates over a grid defined by multiple ranges.
    """

    def __init__(
            self,
            ranges: DimensionRanges,
            direction: EnumerationDirection = ReversedEnumerationDirection()
    ):
        self.ranges = ranges
        self.direction = direction
        self.first = True

    @staticmethod
    def of(ranges, direction: EnumerationDirection = ReversedEnumerationDirection()):
        new = []

        for r in ranges:
            if isinstance(r, list):
                new.append(GridIterator.of(r, direction))
            else:
                new.append(r)

        return GridIterator(new, direction)

    def within(self) -> bool:
        return any([r.within() for r in self.ranges])

    def step(self):
        return self.do_next(self.ranges)

    def __iter__(self):
        return self

    def __next__(self):
        if self.first:
            self.first = False

            return self.get_value()

        return self.step()

    def do_next(self, rs):
        for r in self.direction.direct(rs):

            if r.within():
                r.step()

                return self.get_value()

            r.reset()

        raise StopIteration

    def split(self, n: int) -> list:
        new = []

        for i, r in enumerate(self.ranges):
            for j, s in enumerate(r.split(n)):

                if len(new) < j + 1:
                    new.append([])

                new[j].append(s)

        result = []

        for n in new:
            result.append(GridIterator.of(n))

        return result

    def reset(self):
        for r in self.ranges:
            r.reset()

    def get_value(self) -> list:
        return [r.get_value() for r in self.ranges]

    def get_range(self, idx: int):
        return self.ranges[idx]

    def __str__(self) -> str:
        return 'GridIterator(' + ', '.join([str(r) for r in self.ranges]) + ')'


class EnsureSumGridIterator(GridRange):
    """
    Ensures that the sum of values in a GridIterator is equal to a specified value.
    """

    def __init__(
            self,
            r: GridIterator,
            s: Numeric
    ):
        self.r = r
        self.s = s
        self.first = True

        while not self.within():
            self.r.step()

    def get_value(self) -> T:
        return self.r.get_value()

    def within(self) -> bool:
        return math.isclose(EnsureSumGridIterator.sum_deep(self.get_value()), self.s)

    def step(self) -> T:
        if not self.first:
            self.r.step()

        if self.first:
            self.first = False

        while not self.within():
            self.r.step()

        return self.r.get_value()

    def reset(self):
        self.r.reset()

    def __iter__(self):
        return self

    def __next__(self):
        return self.step()

    @staticmethod
    def sum_deep(value) -> float:
        if isinstance(value, list):
            s = 0

            for v in value:
                s += EnsureSumGridIterator.sum_deep(v)

            return s

        return value

    def split(self, n: int) -> list:
        return [self.r]

class GridIteratorDataGenerator(OptionallyFiniteDataGenerator[T]):
    """
    A data generator that generates data using a GridIterator.
    """

    def __init__(self, iterator: GridIterator):
        super().__init__()
        self.iterator = iterator

    def next(self):
        return self.supply(self.iterator.__next__())

    @abstractmethod
    def supply(self, data: any) -> any:
        pass

class GaussianMixtureDataGenerator(GridIteratorDataGenerator[T]):
    """
    A data generator that generates data for Gaussian mixture models.
    """

    def supply(self, data: any) -> any:
        return data, olr(data[0], data[1], data[2])
