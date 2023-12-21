# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-05-05T16:18:04+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2022-10-03T23:04:17+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from zellij.core.variables import Variable
    from zellij.core.search_space import Searchspace

import logging

logger = logging.getLogger("zellij.addons")


class Addon(ABC):
    """Addon

    Abstract class describing what an addon is.
    An :code:`Addon` in Zellij, is an additionnal feature that can be added to a
    :code:`target` object. See :ref:`varadd` for addon targeting :ref:`var` or
    :ref:`spadd` targeting :ref:`sp`.

    Parameters
    ----------
    target : Object, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : Object, default=None
        Object targeted by the addons

    """

    def __init__(self, target: object = None):
        self.target = target

    @property
    def target(self) -> object:
        return self._target

    @target.setter
    def target(self, object: object):
        self._target = object


class VarAddon(Addon):
    """
    :ref:`addons` where the target must be of type :ref:`var`.

    Parameters
    ----------
    target : :ref:`var`, default=None
        :ref:`var` targeted by the addons

    Attributes
    ----------
    target : :ref:`var`, default=None
        :ref:`var` targeted by the addons

    """

    def __init__(self, target: Optional[Variable] = None):
        super(VarAddon, self).__init__(target)

    @property
    def target(self) -> Optional[Variable]:
        return self._target

    @target.setter
    def target(self, variable: Optional[Variable]):
        from zellij.core.variables import Variable

        if variable:
            assert isinstance(variable, Variable), logger.error(
                f"Object must be a `Variable` for {self.__class__.__name__}, got {variable}"
            )

        self._target = variable


class SearchspaceAddon(Addon):
    """
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    """

    def __init__(self, target: Optional[Searchspace] = None):
        super(SearchspaceAddon, self).__init__(target)

    @property
    def target(self) -> Optional[Searchspace]:
        return self._target

    @target.setter
    def target(self, search_space: Optional[Searchspace]):
        from zellij.core.search_space import Searchspace

        if search_space:
            assert isinstance(search_space, Searchspace), logger.error(
                f"Object must be a `Searchspace` for {self.__class__.__name__},\
                 got {search_space}"
            )

        self._target = search_space


class Neighborhood(SearchspaceAddon):
    """
    :ref:`addons` where the target must be of type :ref:`sp`.
    Describes what a neighborhood is for a :ref:`sp`.

    Parameters
    ----------
    neighborhood : object
        Neighborhood of :code:`target`.
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        Object targeted by the addons

    """

    def __init__(self, neighborhood: object, target: Optional[Searchspace] = None):
        super(Neighborhood, self).__init__(target)
        self.neighborhood = neighborhood  # type: ignore

    @property
    def neighborhood(self) -> object:
        return self._neighborhood  # type: ignore

    @abstractmethod
    def __call__(self, point: object, size: int = 1):
        pass


class VarNeighborhood(VarAddon):
    """
    :ref:`addons` where the target must be of type :ref:`var`.
    Describes what a neighborhood is for a :ref:`var`.

    Parameters
    ----------
    target : :ref:`var`, default=None
        Object targeted by the addons

    Attributes
    ----------
    target : :ref:`var`, default=None
        Object targeted by the addons

    """

    def __init__(self, neighborhood: object, target: Optional[Variable] = None):
        super(VarAddon, self).__init__(target)
        self.neighborhood = neighborhood  # type: ignore

    @property
    def neighborhood(self) -> Optional[Variable]:
        return self._neighborhood  # type: ignore

    @abstractmethod
    def __call__(self, point: object, size: int = 1):
        pass


class Converter(SearchspaceAddon):
    """
    :ref:`addons` where the target must be of type :ref:`sp`.
    Describes what a converter is for a :ref:`sp`.
    Converter allows to convert the type of a :ref:`sp` to another one.
    All :ref:`var` must have a converter :ref:`varadd` implemented.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    """

    def __init__(self, target: Optional[Searchspace] = None):
        super(Converter, self).__init__(target)

    @abstractmethod
    def convert(self) -> object:
        pass

    @abstractmethod
    def reverse(self) -> object:
        pass


class VarConverter(VarAddon):
    """
    :ref:`addons` where the target must be of type :ref:`var`.
    Describes what a converter is for a :ref:`var`.
    Converter allows to convert the type of a :ref:`var` to another one.

    Parameters
    ----------
    target : :ref:`var`, default=None
        :ref:`var` targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        :ref:`var` targeted by the addons

    """

    def __init__(self, target: Optional[Variable] = None):
        super(VarConverter, self).__init__(target)

    @abstractmethod
    def convert(self) -> object:
        pass

    @abstractmethod
    def reverse(self) -> object:
        pass


class Operator(SearchspaceAddon):
    """
    Abstract class describing what an operator is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    """

    def __init__(self, sp: Optional[Searchspace] = None):
        super(Operator, self).__init__(sp)

    @abstractmethod
    def __call__(self):
        pass


class Mutator(SearchspaceAddon):
    """
    Abstract class describing what an Mutator is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    """

    def __init__(self, search_space=None):
        super(Mutator, self).__init__(search_space)


class Crossover(SearchspaceAddon):
    """
    Abstract class describing what an MCrossover is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    """

    def __init__(self, target: Optional[Searchspace] = None):
        super(Crossover, self).__init__(target)


class Selector(SearchspaceAddon):
    """
    Abstract class describing what an Selector is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    Attributes
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    """

    def __init__(self, target: Optional[Searchspace] = None):
        super(Selector, self).__init__(target)


class Distance(SearchspaceAddon):
    """
    Abstract class describing what an Distance is for a :ref:`sp`.
    :ref:`addons` where the target must be of type :ref:`sp`.

    Parameters
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons
    weights : list[float], default=None
        List of floats giving weights for each feature of the :ref:`sp`
    Attributes
    ----------
    target : :ref:`sp`, default=None
        :ref:`sp` targeted by the addons

    """

    def __init__(
        self,
        target: Optional[Searchspace] = None,
        weights: Optional[list[float]] = None,
    ):
        super(Distance, self).__init__(target)
        self.weights = None

    @abstractmethod
    def __call__(self, point_a, point_b):
        pass
