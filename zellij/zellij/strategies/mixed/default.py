# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-04-06T17:28:46+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


from zellij.core.metaheuristic import Metaheuristic
import numpy as np

import logging

logger = logging.getLogger("zellij.Rnd")


class Default(Metaheuristic):

    """Default

    Evaluate a given list of solutions.

    Attributes
    ----------

    search_space : Searchspace
        Search space object containing bounds of the search space.

    solution : list
        List of lists. Each elements represents a single solution.

    batch : int, default=1
        Batch size of returned solution at each :code:`forward`.


    verbose : boolean, default=True
        Algorithm verbosity

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is in Zellij.
    :ref:`lf` : Describes what a loss function is in Zellij.
    :ref:`sp` : Describes what a search space is in Zellij.
    """

    def __init__(
        self,
        search_space,
        solutions,
        batch=1,
        verbose=True,
    ):
        """__init__(search_space, size=1, verbose=True)

        Initialize Genetic_algorithm class

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space.

        solution : list
            List of lists or list of dictionnaries. Each elements represents a single solution.

        batch : int, default=1
            Batch size of returned solution at each :code:`forward`.

        verbose : boolean, default=True
            Algorithm verbosity

        """

        super().__init__(search_space, verbose)

        ##############
        # PARAMETERS #
        ##############

        self.batch = batch
        self.solutions = solutions * self.batch

        # if batch > len(self.solutions):
        #     if isinstance(self.solutions[0], list):
        #         self.batch_sol = {
        #             "batch_0": np.resize(
        #                 self.solutions, (self.batch, self.search_space.size)
        #             )
        #         }
        #     else:
        #         raise ValueError(
        #             f"Unknown type of solution, must be a list of lists. Got {type(self.solutions[0])}"
        #         )
        # elif batch < len(self.solutions):
        #     nbatches = int(np.ceil(len(solutions) / batch))
        #     self.batch_sol = {}
        #     for i in range(nbatches - 1):
        #         self.batch_sol[f"batch_{i}"] = self.solutions[
        #             i * batch : i * batch + batch
        #         ]

        #     i = nbatches - 1
        #     self.batch_sol[f"batch_{i}"] = self.solutions[i * batch : i * batch + batch]
        #     add = len(solutions) % batch
        #     if add > 0:
        #         self.batch_sol[f"batch_{i}"].extend(self.solutions[:add])
        # else:
        #     self.batch_sol = self.solutions

        #############
        # VARIABLES #
        #############
        # iterations
        self.i = 0

    # Run
    def forward(self, X, Y, constraints):
        """forward(X, Y)
        Runs one step of Random.

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

        logger.info("GA Starting")

        # batch = self.i % len(self.batch_sol)
        # solutions = self.batch_sol[f"batch_{batch}"]
        infos = {"algorithm": "Default", "iteration": self.i}
        self.i += 1

        return self.solutions, infos
