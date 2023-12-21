# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-01-19T19:21:45+01:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np
from zellij.core.metaheuristic import ContinuousMetaheuristic
from zellij.core.search_space import Fractal, ContinuousSearchspace

import logging

logger = logging.getLogger("zellij.ILS")


# Intensive local search
class ILS(ContinuousMetaheuristic):

    """ILS

    Intensive local search is an exploitation algorithm comming from the
    original FDA paper.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.
    red_rate : float, default=0.5
        Determines the step reduction rate each time an improvement happens.
    precision : flaot, default=1e-20
        When :code:`step`<:code:`precision`, stops the algorithm.
    verbose : boolean, default=True
        Activate or deactivate the progress bar.


    Methods
    -------

    forward(X, Y)
        Runs one step of ILS


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space,
        red_rate=0.5,
        inflation=1.75,
        verbose=True,
    ):
        """__init__(search_space, red_rate=0.5,verbose=True)

        Initialize ILS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        red_rate : float, default=0.5
            Determines the step reduction rate each time an improvement happens.
        inflation : float, default=1.75
            Inflation rate of the :code:`Hypersphere`.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        self.red_rate = red_rate
        self.inflation = inflation
        super().__init__(search_space, verbose)

        #############
        # VARIABLES #
        #############

        self.initialized = False
        self.current_dim = 0
        self.current_score = float("inf")
        self.current_point = np.zeros(self.search_space.size)
        self.improvement = False

    @ContinuousMetaheuristic.search_space.setter
    def search_space(self, value):
        if value:
            if (
                isinstance(value, ContinuousSearchspace)
                or isinstance(value, Fractal)
                or hasattr(value, "converter")
            ):
                self._search_space = value
            else:
                raise ValueError(
                    f"Search space must be continuous, a fractal or have a `converter` addon, got {value}"
                )

            if not (hasattr(value, "lower") and hasattr(value, "upper")):
                raise AttributeError(
                    "Search space must have lower and upper bounds attributes, got {value}."
                )

            if hasattr(value, "center"):
                self.center = value.center  # type: ignore
            else:
                self.center = (value.upper + value.lower) / 2

            if hasattr(value, "radius"):
                self.radius = value.radius  # type: ignore
            else:
                self.radius = ((value.upper - value.lower) / 2)[0]

            self.step = self.radius * self.inflation

    def reset(self):
        """reset()

        Reset ILS variables to their initial values.

        """
        super().reset()

        self.step = np.tile(self.radius * self.inflation, 2)
        self.step[1] = -self.step[1]

        self.initialized = False
        self.current_dim = 0
        self.current_score = float("inf")
        self.current_point = np.zeros(self.search_space.size)
        self.improvement = False

    def _one_step(self):
        points = np.tile(self.current_point, (2, 1))
        points[0, self.current_dim] += self.step
        points[1, self.current_dim] -= self.step
        points[:, self.current_dim] = np.clip(points[:, self.current_dim], 0.0, 1.0)

        return points, {"algorithm": "ILS"}

    def forward(self, X, Y, constraint=None):
        """forward(X, Y)
        Runs one step of ILS.

        Parameters
        ----------
        X : list
            List of previously computed points
        Y : list
            List of loss value linked to :code:`X`.
            :code:`X` and :code:`Y` must have the same length.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        # logging
        logger.info("Starting")

        if self.initialized:
            argmin = np.argmin(Y)

            if Y[argmin] < self.current_score:
                self.current_score = Y[argmin]
                self.current_point = X[argmin]
                self.improvement = True
        else:
            self.current_point = self.center
            self.initialized = True

        if self.current_dim >= self.search_space.size:
            self.current_dim = 0
            if self.improvement:
                self.improvement = False
            else:
                self.step *= self.red_rate

            logger.debug(f"Evaluating dimension {self.current_dim}")
            logger.info("Ending")

            return self._one_step()

        else:
            to_return = self._one_step()
            self.current_dim += 1

            logger.debug(f"Evaluating dimension {self.current_dim}")
            logger.info("Ending")

            return to_return


# Intensive local search
class ILS_section(ILS):

    """ILS_section

    Intensive local search is an exploitation algorithm comming from the
    original FDA paper. It evaluates a point in each dimension arround
    an initial solution. Distance of the computed point to the initial one is
    decreasing according to a reduction rate. At each iteration the algorithm
    moves to the best solution found.

    This variation works with :code:`Section` fractals.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.

    red_rate : float, default=0.5
        Determines the step reduction rate each time an improvement happens.

    verbose : boolean, default=True
        Activate or deactivate the progress bar.


    Methods
    -------

    forward(X, Y)
        Runs one step of ILS.


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space,
        red_rate=0.5,
        verbose=True,
    ):
        """__init__(search_space, red_rate=0.5,verbose=True)

        Initialize ILS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        red_rate : float, default=0.5
            Determines the step reduction rate each time an improvement happens.

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, red_rate, 1, verbose)

        self.length = np.max(self.search_space.upper - self.search_space.lower)
        self.step = np.tile(self.length, 2)
        self.step[1] = -self.step[1]

    @ContinuousMetaheuristic.search_space.setter
    def search_space(self, value):
        if value:
            if (
                isinstance(value, ContinuousSearchspace)
                or isinstance(value, Fractal)
                or hasattr(value, "converter")
            ):
                self._search_space = value
            else:
                raise ValueError(
                    f"Search space must be continuous, a fractal or have a `converter` addon, got {value}"
                )

            if not (hasattr(value, "lower") and hasattr(value, "upper")):
                raise AttributeError(
                    "Search space must have lower and upper bounds attributes, got {value}."
                )
        if value:
            self.center = (value.upper + value.lower) / 2
            self.radius = np.max(value.upper - value.lower)
            self.step = np.tile(self.radius * self.inflation, 2)
            self.step[1] = -self.step[1]

    def reset(self):
        """reset()

        Reset ILS variables to their initial values.

        """

        self.step = np.tile(self.radius, 2)
        self.step[1] = -self.step[1]

        self.initialized = False
        self.current_dim = 0
        self.current_score = float("inf")
        self.current_point = np.zeros(self.search_space.size)
        self.improvement = False


class ILS_random(ContinuousMetaheuristic):

    """ILS_random

    Intensive local search is an exploitation algorithm comming from the
    original FDA paper. Modified to sample :code:`n` symmetric points on the borders, and symmetrize them.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.
    red_rate : float, default=0.5
        Determines the step reduction rate each time an improvement happens.
    precision : flaot, default=1e-20
        When :code:`step`<:code:`precision`, stops the algorithm.
    verbose : boolean, default=True
        Activate or deactivate the progress bar.


    Methods
    -------

    forward(X, Y)
        Runs one step of ILS


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space,
        red_rate=0.5,
        inflation=1.75,
        verbose=True,
    ):
        """__init__(search_space, red_rate=0.5,verbose=True)

        Initialize ILS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        red_rate : float, default=0.5
            Determines the step reduction rate each time an improvement happens.
        inflation : float, default=1.75
            Inflation rate of the :code:`Hypersphere`.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        self.red_rate = red_rate
        self.inflation = inflation
        super().__init__(search_space, verbose)

        #############
        # VARIABLES #
        #############

        self.initialized = False
        self.current_dim = 0
        self.current_score = float("inf")
        self.current_point = np.zeros(self.search_space.size)
        self.improvement = False

    @ContinuousMetaheuristic.search_space.setter
    def search_space(self, value):
        if value:
            if (
                isinstance(value, ContinuousSearchspace)
                or isinstance(value, Fractal)
                or hasattr(value, "converter")
            ):
                self._search_space = value
            else:
                raise ValueError(
                    f"Search space must be continuous, a fractal or have a `converter` addon, got {value}"
                )

            if not (hasattr(value, "lower") and hasattr(value, "upper")):
                raise AttributeError(
                    "Search space must have lower and upper bounds attributes, got {value}."
                )

            if hasattr(value, "center"):
                self.center = value.center  # type: ignore
            else:
                self.center = (value.upper + value.lower) / 2

            if hasattr(value, "radius"):
                self.radius = value.radius  # type: ignore
            else:
                self.radius = ((value.upper - value.lower) / 2)[0]

            self.step = self.radius * self.inflation

    def reset(self):
        """reset()

        Reset ILS variables to their initial values.

        """

        self.step = self.radius * self.inflation

        self.initialized = False
        self.current_dim = 0
        self.current_score = float("inf")
        self.current_point = np.zeros(self.search_space.size)
        self.improvement = False

    def _one_step(self):
        points = np.tile(self.current_point, (2 * self.search_space.size, 1))

        points[
            : self.search_space.size
        ] = self.search_space.lower + self.step * np.random.random(
            (self.search_space.size, self.search_space.size)
        ) * (
            self.search_space.upper - self.search_space.lower
        )

        points[self.search_space.size :] = (
            self.search_space.upper
            + self.search_space.lower
            - points[: self.search_space.size]
        )

        return points, {"algorithm": "ILS_random"}

    def forward(self, X, Y, constraint=None):
        """forward(X, Y)
        Runs one step of ILS.

        Parameters
        ----------
        X : list
            List of previously computed points
        Y : list
            List of loss value linked to :code:`X`.
            :code:`X` and :code:`Y` must have the same length.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        # logging
        logger.info("Starting")

        if self.initialized:
            argmin = np.argmin(Y)

            if Y[argmin] < self.current_score:
                self.current_score = Y[argmin]
                self.current_point = X[argmin]
                self.improvement = True
        else:
            self.current_point = self.center
            self.initialized = True

        if self.current_dim >= self.search_space.size:
            self.current_dim = 0
            if self.improvement:
                self.improvement = False
            else:
                self.step *= self.red_rate

            logger.debug(f"Evaluating dimension {self.current_dim}")
            logger.info("Ending")

            return self._one_step()

        else:
            to_return = self._one_step()
            self.current_dim += 1

            logger.debug(f"Evaluating dimension {self.current_dim}")
            logger.info("Ending")

            return to_return


class ILS_sym(ContinuousMetaheuristic):

    """ILS_sym

    Intensive local search is an exploitation algorithm comming from the
    original FDA paper. Modified to sample :code:`n` fixed and symmetric points on the borders, and symmetrize them.

    Attributes
    ----------
    search_space : Searchspace
        :ref:`sp` object containing decision variables and the loss function.
    red_rate : float, default=0.5
        Determines the step reduction rate each time an improvement happens.
    precision : flaot, default=1e-20
        When :code:`step`<:code:`precision`, stops the algorithm.
    verbose : boolean, default=True
        Activate or deactivate the progress bar.


    Methods
    -------

    forward(X, Y)
        Runs one step of ILS


    See Also
    --------
    Metaheuristic : Parent class defining what a Metaheuristic is.
    LossFunc : Describes what a loss function is in Zellij
    Searchspace : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space,
        red_rate=0.5,
        inflation=1.75,
        verbose=True,
    ):
        """__init__(search_space, red_rate=0.5,verbose=True)

        Initialize ILS class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.
        red_rate : float, default=0.5
            Determines the step reduction rate each time an improvement happens.
        inflation : float, default=1.75
            Inflation rate of the :code:`Hypersphere`.
        verbose : boolean, default=True
            Algorithm verbosity

        """

        self.red_rate = red_rate
        self.inflation = inflation
        super().__init__(search_space, verbose)

        #############
        # VARIABLES #
        #############

        self.initialized = False
        self.current_dim = 0
        self.current_score = float("inf")
        self.current_point = np.zeros(self.search_space.size)
        self.improvement = False

    @ContinuousMetaheuristic.search_space.setter
    def search_space(self, value):
        if value:
            if (
                isinstance(value, ContinuousSearchspace)
                or isinstance(value, Fractal)
                or hasattr(value, "converter")
            ):
                self._search_space = value
            else:
                raise ValueError(
                    f"Search space must be continuous, a fractal or have a `converter` addon, got {value}"
                )

            if not (hasattr(value, "lower") and hasattr(value, "upper")):
                raise AttributeError(
                    "Search space must have lower and upper bounds attributes, got {value}."
                )

            if hasattr(value, "center"):
                self.center = value.center  # type: ignore
            else:
                self.center = (value.upper + value.lower) / 2

            if hasattr(value, "radius"):
                self.radius = value.radius  # type: ignore
            else:
                self.radius = ((value.upper - value.lower) / 2)[0]

            self.step = self.radius * self.inflation

    def reset(self):
        """reset()

        Reset ILS variables to their initial values.

        """

        self.step = self.radius * self.inflation

        self.initialized = False
        self.current_dim = 0
        self.current_score = float("inf")
        self.current_point = np.zeros(self.search_space.size)
        self.improvement = False

    def _one_step_i3(self):
        points = np.zeros((2 * self.search_space.size, self.search_space.size))

        print(points.shape)

        lower = np.maximum(self.current_point - self.radius, 0.0)
        upper = np.minimum(self.current_point + self.radius, 1.0)

        center = (upper + lower) / 2

        # X
        X = lower + np.random.random(self.search_space.size) * (upper - lower)
        points[0] = X[:]
        # -X
        points[1] = -X + center

        # Xh - Xps
        points[2] = X[:]
        points[2, 0] = -X[0] + center[0]
        # Xp - XH
        points[3] = np.copy(points[1])
        points[3, 1] = X[1]

        points = np.clip(points, 0.0, 1.0)
        print(points)

        return points, {"algorithm": "ILS_random"}

    def _one_step_s3(self):
        points = np.zeros((4 * self.search_space.size, self.search_space.size))

        lower = np.maximum(self.current_point - self.radius, 0.0)
        upper = np.minimum(self.current_point + self.radius, 1.0)

        center = (upper + lower) / 2

        for rdim in range(self.search_space.size):
            cidx = rdim * 4
            # X
            X = lower + np.random.random(self.search_space.size) * (upper - lower)
            points[cidx : cidx + 4] = np.tile(X, (4, 1))
            # -X
            points[cidx + 1] = -points[cidx + 1] + center
            # Xh - Xps
            points[cidx + 2, rdim] = -points[cidx + 2, rdim] + center[rdim]
            # Xp - XH
            points[cidx + 3] = -points[cidx + 3]
            points[cidx + 3, rdim] = -points[cidx + 3, rdim] + center[rdim]

        points = np.clip(points, 0.0, 1.0)

        return points, {"algorithm": "ILS_random"}

    def forward(self, X, Y, constraint=None):
        """forward(X, Y)
        Runs one step of ILS.

        Parameters
        ----------
        X : list
            List of previously computed points
        Y : list
            List of loss value linked to :code:`X`.
            :code:`X` and :code:`Y` must have the same length.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        # logging
        logger.info("Starting")

        if self.initialized:
            argmin = np.argmin(Y)

            if Y[argmin] < self.current_score:
                self.current_score = Y[argmin]
                self.current_point = X[argmin]
                self.improvement = True
        else:
            self.current_point = self.center
            self.initialized = True

        if self.current_dim >= self.search_space.size:
            self.current_dim = 0
            if self.improvement:
                self.improvement = False
            else:
                self.step *= self.red_rate

            logger.debug(f"Evaluating dimension {self.current_dim}")
            logger.info("Ending")

            if self.search_space.size > 3:
                return self._one_step_s3()
            else:
                return self._one_step_i3()

        else:
            if self.search_space.size > 3:
                to_return = self._one_step_s3()
            else:
                to_return = self._one_step_i3()

            self.current_dim += 1

            logger.debug(f"Evaluating dimension {self.current_dim}")
            logger.info("Ending")

            return to_return
