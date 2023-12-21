# @Author: Thomas Firmin <tfirmin>
# @Date:   2023-02-22T15:37:46+01:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-03-31T20:30:26+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

import torch
import numpy as np

from abc import ABC, abstractmethod


class EarlyStopping(ABC):
    """EarlyStopping

    Abstract method. `__call__` method returns:
    - `True` if network has to be stopped
    - Else `False`

    """

    def __init__(self):
        super(EarlyStopping, self).__init__()

    @abstractmethod
    def __call__(self, objective, network):
        pass

    def __and__(self, other):
        return AndStopping(self, other)

    def __or__(self, other):
        return OrStopping(self, other)

    def reset(self):
        pass


class PatienceStopping(EarlyStopping):
    """PatienceStopping

    Class describing an Early stopping based on patience.
    Return True if the stopping criterion is True after
    :math:`n` calls to this class.

    """

    def __init__(self, patience, do_reset=False):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.do_reset = do_reset

        self._counter = 0

    def get_inequality(self):
        return

    def __call__(self, condition, objective, network):
        print("PATIENCE MY BROTHER")
        if condition(objective, network):
            self._counter += 1
        else:
            self._counter = 0

        if self._counter >= self.patience:
            self._counter = 0
            return True
        else:
            return False

    def __and__(self, other):
        return AndStopping(self, other)

    def __or__(self, other):
        return OrStopping(self, other)

    def reset(self):
        super(EarlyStopping).reset()
        self._counter = 0


class AndStopping(EarlyStopping):
    """AndStopping

    Describes combination of early stopping.
    Return results from es1(...) & es2(...)
    """

    def __init__(self, es1, es2):
        """
        Parameters
        ----------
        es1 : EarlyStopping
            EarlyStopping object 1
        es2 : EarlyStopping
            EarlyStopping object 2

        Returns
        -------
        bool
            Return result from es1(...) & es2(...)

        """
        super(AndStopping, self).__init__()
        self.es1 = es1
        self.es2 = es2

    def __call__(self, objective, network):
        return self.es1(objective, network) & self.es2(objective, network)

    def reset(self):
        self.es1.reset()
        self.es2.reset()

    def to_zeroinequality(self):
        return [self.es1.to_zeroinequality(), self.es2.to_zeroinequality()]


class OrStopping(EarlyStopping):
    """OrStopping

    Describes combination of early stopping.
    Return results from es1(...) | es2(...)
    """

    def __init__(self, es1, es2):
        """
        Parameters
        ----------
        es1 : EarlyStopping
            EarlyStopping object 1
        es2 : EarlyStopping
            EarlyStopping object 2

        Returns
        -------
        bool
            Return result from es1(...) & es2(...)

        """
        super(OrStopping, self).__init__()
        self.es1 = es1
        self.es2 = es2

    def __call__(self, objective, network):
        return self.es1(objective, network) | self.es2(objective, network)

    def reset(self):
        self.es1.reset()
        self.es2.reset()

    def to_zeroinequality(self):
        return [self.es1.to_zeroinequality(), self.es2.to_zeroinequality()]


class SpikeStopping(PatienceStopping):
    """SpikeStopping

    Abstract class for early stopping based on spikes.

    Parameters
    ----------
    layer : type
        Description of parameter `layer`.
    condition : Callable
        Callable taking into parameter number of input spikes
    patience : int, default=1
        Return True if the stopping criterion is True after
        :math:`n` calls to this class.
    """

    def __init__(self, layer, condition, patience=1):
        super(SpikeStopping, self).__init__(patience)
        self.layer = layer
        self.condition = condition

    def __call__(self, objective, network):
        return super(SpikeStopping, self).__call__(self.condition, objective, network)


class SpikePerDatumStopping(SpikeStopping):
    def __init__(self, layer, threshold, patience=1):
        self.threshold = threshold
        condition = (
            lambda objective, network: network.recorders_train[self.layer]
            / network.computed_images_train
            < threshold
        )
        super(SpikePerDatumStopping, self).__init__(layer, condition, patience)

    def __call__(self, objective, network):
        self._last_zero = self._compute_zineq(objective, network)
        return super(SpikePerDatumStopping, self).__call__(objective, network)

    def _compute_zineq(self, objective, network):
        (
            -network.recorders_train[self.layer] / network.computed_images_train
            + self.threshold
        )

    def to_zeroinequality(self):
        return self._last_zero

    def reset(self):
        super().reset()
        self._last_zero = 0


class NonSpikingPercentageStopping(EarlyStopping):
    def __init__(self, layer, threshold, proportion, total):
        super(NonSpikingPercentageStopping, self).__init__()
        self.layer = layer
        self.threshold = threshold
        self.proportion = proportion
        self.total = total

        self.count = 0

    def __call__(self, objective, network):
        nsimg = np.count_nonzero(
            np.array(objective.img_spikes[self.layer]) < self.threshold
        )
        self.count += nsimg
        print(
            f"COUNT: {self.count}/{self.total}>{self.proportion}, PROCESSED: {network.computed_images_train}"
        )

        return self.count / self.total > self.proportion

    def reset(self):
        self.count = 0
        super().reset()

    def to_zeroinequality(self):
        return self.count / self.total - self.proportion
