# @Author: Thomas Firmin <tfirmin>
# @Date:   2023-04-21T11:19:42+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-19T18:00:55+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from zellij.core import ArrayVar, FloatVar, IntVar, CatVar
from zellij.utils.neighborhoods import (
    FloatInterval,
    ArrayInterval,
    IntInterval,
    CatInterval,
)
from zellij.utils.converters import FloatMinmax, IntMinmax, CatMinmax, ArrayConverter
import scipy
import numpy as np


def loguniform(low, high, size=None):
    return scipy.stats.loguniform.rvs(low, high, size=size)


# reverse
def rev_loguniform(low, high, size=None):
    return high - scipy.stats.loguniform.rvs(low, high, size=size) + low


# discrete loguniform
def disc_loguniform(low, high, size=None):
    return np.round(scipy.stats.loguniform.rvs(low, high, size=size)).astype(int)


# reverse
def disc_rev_loguniform(low, high, size=None):
    return np.round(
        high - scipy.stats.loguniform.rvs(low, high, size=size) + low
    ).astype(int)


def get_gendh_rate_2():
    values = ArrayVar("", [], neighbor=ArrayInterval())

    values.append(CatVar("decoder", ["all", "vote", "2gram"], neighbor=CatInterval()))
    values.append(IntVar("epochs", 1, 3, neighbor=IntInterval(1)))
    # values.append(
    #    IntVar("batch_size", 1, 50, neighbor=IntInterval(10), sampler=disc_loguniform)
    # )

    # Diehl and cook
    values.append(IntVar("map_size", 20, 2000, neighbor=IntInterval(50)))

    ## STDP
    values.append(FloatVar("nu_pre", 1e-4, 1e-2, neighbor=FloatInterval(5e-4)))
    values.append(FloatVar("nu_post", 1e-4, 1e-2, neighbor=FloatInterval(5e-3)))
    values.append(FloatVar("strength_exc", 250, 500, neighbor=FloatInterval(10)))
    values.append(FloatVar("strength_inh", 250, 500, neighbor=FloatInterval(10)))
    values.append(FloatVar("norm", 50, 200, neighbor=FloatInterval(10)))

    ## Excitatory layer
    values.append(FloatVar("e_thresh", -59, 0, neighbor=FloatInterval(6)))
    values.append(FloatVar("e_rest", -70, -60, neighbor=FloatInterval(1)))
    values.append(IntVar("e_refrac", 0, 20, neighbor=IntInterval(2)))
    values.append(FloatVar("theta_plus", 0.001, 0.5, neighbor=FloatInterval(0.01)))
    values.append(FloatVar("tc_theta_decay", 1e6, 2e7, neighbor=FloatInterval(5e6)))
    values.append(FloatVar("e_tc_decay", 5, 5000, neighbor=FloatInterval(100)))
    values.append(FloatVar("e_tc_trace", 1, 100, neighbor=IntInterval(5)))

    ## Inhibitory layer
    values.append(FloatVar("i_thresh", -40, 0, neighbor=FloatInterval(4)))
    values.append(FloatVar("i_rest", -60, -45, neighbor=FloatInterval(1.5)))
    values.append(IntVar("i_refrac", 0, 20, neighbor=IntInterval(2)))
    values.append(FloatVar("i_tc_decay", 5, 5000, neighbor=FloatInterval(100)))
    values.append(FloatVar("i_tc_trace", 1, 100, neighbor=IntInterval(5)))

    return values


def get_gendh_rate_2_cnt():
    values = ArrayVar("", [], converter=ArrayConverter())

    values.append(CatVar("decoder", ["all", "vote", "2gram"], converter=CatMinmax()))
    values.append(IntVar("epochs", 1, 3, converter=IntMinmax()))
    # values.append(
    #    IntVar("batch_size", 1, 50, converter=IntMinmax(), sampler=disc_loguniform)
    # )

    # Diehl and cook
    values.append(IntVar("map_size", 10, 1000, converter=IntMinmax()))

    ## STDP
    values.append(
        FloatVar("nu_pre", 1e-4, 1e-3, converter=FloatMinmax(), sampler=loguniform)
    )
    values.append(
        FloatVar("nu_post", 1e-3, 1e-2, converter=FloatMinmax(), sampler=rev_loguniform)
    )
    values.append(
        FloatVar(
            "strength_exc", 50, 500, converter=FloatMinmax(), sampler=rev_loguniform
        )
    )
    values.append(
        FloatVar(
            "strength_inh", 60, 600, converter=FloatMinmax(), sampler=rev_loguniform
        )
    )
    values.append(
        FloatVar("norm", 78.4, 200, converter=FloatMinmax(), sampler=rev_loguniform)
    )

    ## Excitatory layer
    values.append(FloatVar("e_thresh", -70, -50, converter=FloatMinmax()))
    values.append(FloatVar("e_rest", -65, -55, converter=FloatMinmax()))
    values.append(IntVar("e_refrac", 1, 10, converter=IntMinmax()))
    values.append(
        FloatVar("theta_plus", 0.01, 0.1, converter=FloatMinmax(), sampler=loguniform)
    )
    values.append(
        FloatVar(
            "tc_theta_decay", 1e6, 1e7, converter=FloatMinmax(), sampler=rev_loguniform
        )
    )
    values.append(
        FloatVar(
            "e_tc_decay", 1000, 5000, converter=FloatMinmax(), sampler=rev_loguniform
        )
    )

    ## Inhibitory layer
    values.append(FloatVar("i_thresh", -30, -10, converter=FloatMinmax()))
    values.append(FloatVar("i_rest", -60, -40, converter=FloatMinmax()))
    values.append(IntVar("i_refrac", 15, 35, converter=IntMinmax()))
    values.append(
        FloatVar("i_tc_decay", 1000, 5000, converter=FloatMinmax(), sampler=loguniform)
    )

    return values


def get_gendh_rate_2_dvs():
    values = ArrayVar("", [], converter=ArrayConverter())

    values.append(
        CatVar(
            "decoder",
            ["all", "vote", "2gram", "log", "svm"],
            converter=CatMinmax(),
        )
    )
    values.append(IntVar("epochs", 1, 12, converter=IntMinmax()))
    # values.append(
    #    IntVar("batch_size", 1, 50, converter=IntMinmax(), sampler=disc_loguniform)
    # )

    # Diehl and cook
    values.append(IntVar("map_size", 20, 1000, converter=IntMinmax()))

    ## STDP
    values.append(
        FloatVar("nu_pre", 1e-4, 1e-2, converter=FloatMinmax(), sampler=loguniform)
    )
    values.append(
        FloatVar("nu_post", 1e-4, 1e-2, converter=FloatMinmax(), sampler=rev_loguniform)
    )

    values.append(
        FloatVar("strength_exc", 1, 500, converter=FloatMinmax(), sampler=loguniform)
    )
    values.append(
        FloatVar("strength_inh", 1, 500, converter=FloatMinmax(), sampler=loguniform)
    )

    # values.append(
    #    FloatVar("norm", 78.4, 784, converter=FloatMinmax(), sampler=loguniform)
    # )

    ## Excitatory layer
    values.append(FloatVar("e_thresh", -59, 60, converter=FloatMinmax()))
    values.append(FloatVar("e_rest", -140, -60, converter=FloatMinmax()))
    values.append(IntVar("e_refrac", 0, 40, converter=IntMinmax()))
    values.append(
        FloatVar("theta_plus", 0.01, 1, converter=FloatMinmax(), sampler=rev_loguniform)
    )

    values.append(
        FloatVar(
            "tc_theta_decay",
            1e6,
            2e7,
            converter=FloatMinmax(),
            sampler=rev_loguniform,
        )
    )
    values.append(
        FloatVar("e_tc_decay", 1, 5000, converter=FloatMinmax()), sampler=loguniform
    )
    values.append(
        FloatVar("e_tc_trace", 1, 5000, converter=FloatMinmax()), sampler=loguniform
    )

    ## Inhibitory layer
    values.append(FloatVar("i_thresh", -40, 40, converter=FloatMinmax()))
    values.append(FloatVar("i_rest", -120, -45, converter=FloatMinmax()))
    values.append(IntVar("i_refrac", 0, 40, converter=IntMinmax()))
    values.append(
        FloatVar("i_tc_decay", 1, 5000, converter=FloatMinmax()), sampler=loguniform
    )
    values.append(
        FloatVar("i_tc_trace", 1, 5000, converter=FloatMinmax()), sampler=loguniform
    )
    return values


def get_gendh_rate_2_cnt_dvs():
    values = ArrayVar("", [], converter=ArrayConverter())

    values.append(
        CatVar(
            "decoder",
            ["all", "vote", "2gram", "log", "svm"],
            converter=CatMinmax(),
        )
    )
    # values.append(IntVar("epochs", 1, 3, converter=IntMinmax()))
    # values.append(
    #    IntVar("batch_size", 1, 50, converter=IntMinmax(), sampler=disc_loguniform)
    # )
    values.append(
        IntVar("reset_interval", 10, 100, converter=IntMinmax(), sampler=rev_loguniform)
    )

    # Diehl and cook
    values.append(IntVar("map_size", 50, 500, converter=IntMinmax()))

    ## STDP
    values.append(
        FloatVar("nu_pre", 1e-3, 1e-2, converter=FloatMinmax(), sampler=loguniform)
    )
    values.append(
        FloatVar("nu_post", 1e-2, 1e-1, converter=FloatMinmax(), sampler=loguniform)
    )

    values.append(
        FloatVar("strength_exc", 10, 160, converter=FloatMinmax(), sampler=loguniform)
    )
    values.append(
        FloatVar("strength_inh", 30, 280, converter=FloatMinmax(), sampler=loguniform)
    )

    values.append(
        FloatVar("norm", 3276.8, 16384, converter=FloatMinmax(), sampler=rev_loguniform)
    )

    ## Excitatory layer
    values.append(FloatVar("e_thresh", -40, 0, converter=FloatMinmax()))
    values.append(FloatVar("e_rest", -140, -75, converter=FloatMinmax()))
    values.append(IntVar("e_refrac", 0, 20, converter=IntMinmax()))
    values.append(
        FloatVar("theta_plus", 0.5, 5, converter=FloatMinmax(), sampler=loguniform)
    )

    values.append(
        FloatVar(
            "tc_theta_decay", 5e6, 5e7, converter=FloatMinmax(), sampler=rev_loguniform
        )
    )
    values.append(
        FloatVar("e_tc_decay", 1000, 5000, converter=FloatMinmax(), sampler=loguniform)
    )
    values.append(
        FloatVar("e_tc_trace", 500, 5000, converter=FloatMinmax(), sampler=loguniform)
    )

    ## Inhibitory layer
    values.append(FloatVar("i_thresh", -10, 10, converter=FloatMinmax()))
    values.append(FloatVar("i_rest", -80, -60, converter=FloatMinmax()))
    values.append(IntVar("i_refrac", 20, 40, converter=IntMinmax()))

    values.append(
        FloatVar("i_tc_decay", 500, 5000, converter=FloatMinmax(), sampler=loguniform)
    )

    return values


######################################
################ LAVA ################
######################################


def get_mnist_rate_lava_cnt():
    values = ArrayVar("", [], converter=ArrayConverter())

    values.append(IntVar("epochs", 1, 40, converter=IntMinmax()))

    values.append(
        IntVar("batch_size", 1, 50, converter=IntMinmax(), sampler=disc_rev_loguniform)
    )

    values.append(CatVar("decoder", ["rate", "max"], converter=CatMinmax()))

    # Architecture
    values.append(IntVar("c1_filters", 1, 128, converter=IntMinmax()))
    values.append(IntVar("c2_filters", 1, 128, converter=IntMinmax()))

    values.append(IntVar("c1_k", 4, 12, converter=IntMinmax()))
    values.append(IntVar("c2_k", 4, 12, converter=IntMinmax()))

    ## SLAYER
    values.append(
        FloatVar(
            "learning_rate", 0.001, 0.1, converter=FloatMinmax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar("tau_grad", 0.1, 1, converter=FloatMinmax(), sampler=loguniform)
    )
    values.append(
        FloatVar("scale_grad", 0.5, 1, converter=FloatMinmax(), sampler=loguniform)
    )

    ## Neurons params

    values.append(FloatVar("threshold", 0.4, 4, converter=FloatMinmax()))

    values.append(
        FloatVar(
            "threshold_step", 0.001, 0.25, converter=FloatMinmax(), sampler=loguniform
        )
    )

    values.append(FloatVar("current_decay", 0.1, 0.99, converter=FloatMinmax()))
    values.append(
        FloatVar(
            "voltage_decay", 0.01, 0.2, converter=FloatMinmax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar(
            "threshold_decay",
            0.01,
            0.5,
            converter=FloatMinmax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            "refractory_decay",
            0.1,
            0.99,
            converter=FloatMinmax(),
            sampler=rev_loguniform,
        )
    )

    values.append(
        FloatVar("dropout", 0.01, 0.90, converter=FloatMinmax(), sampler=loguniform)
    )

    return values


def get_dvs_rate_lava_cnt():
    values = ArrayVar("", [], converter=ArrayConverter())

    values.append(IntVar("epochs", 1, 15, converter=IntMinmax()))

    values.append(
        IntVar("batch_size", 1, 5, converter=IntMinmax(), sampler=disc_rev_loguniform)
    )

    values.append(CatVar("decoder", ["rate", "max"], converter=CatMinmax()))

    # Architecture
    values.append(IntVar("c1_filters", 1, 36, converter=IntMinmax()))
    values.append(IntVar("c2_filters", 1, 36, converter=IntMinmax()))

    values.append(IntVar("c1_k", 4, 48, converter=IntMinmax()))
    values.append(IntVar("c2_k", 4, 48, converter=IntMinmax()))

    ## SLAYER
    values.append(
        FloatVar(
            "learning_rate", 0.001, 0.1, converter=FloatMinmax(), sampler=loguniform
        )
    )
    values.append(
        FloatVar("tau_grad", 0.1, 1, converter=FloatMinmax(), sampler=loguniform)
    )
    values.append(
        FloatVar("scale_grad", 0.5, 1, converter=FloatMinmax(), sampler=loguniform)
    )
    ## Neurons params

    values.append(
        FloatVar("threshold", 0.1, 1, converter=FloatMinmax(), sampler=loguniform)
    )

    values.append(
        FloatVar(
            "threshold_step", 0.001, 0.40, converter=FloatMinmax(), sampler=loguniform
        )
    )

    values.append(
        FloatVar(
            "current_decay", 0.1, 0.99, converter=FloatMinmax(), sampler=rev_loguniform
        )
    )
    values.append(
        FloatVar(
            "voltage_decay", 0.01, 0.9, converter=FloatMinmax(), sampler=rev_loguniform
        )
    )
    values.append(
        FloatVar(
            "threshold_decay",
            0.01,
            0.5,
            converter=FloatMinmax(),
            sampler=loguniform,
        )
    )
    values.append(
        FloatVar(
            "refractory_decay", 0.1, 0.99, converter=FloatMinmax(), sampler=loguniform
        )
    )

    values.append(
        FloatVar("dropout", 0.01, 0.90, converter=FloatMinmax(), sampler=loguniform)
    )
    return values
