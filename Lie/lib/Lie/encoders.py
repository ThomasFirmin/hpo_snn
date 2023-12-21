# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-09-21T16:31:34+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-12T12:56:47+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import numpy as np
import torch
from scipy.signal import firwin
import abc
from bindsnet.encoding.encoders import Encoder


def single(
    datum: torch.Tensor,
    time: int,
    min: float,
    max: float,
    tau: float = 0.1,
    dt: float = 1.0,
    device="cpu",
    **kwargs,
) -> torch.Tensor:
    # language=rst
    """
    Generates timing based single-spike encoding. Spike occurs earlier if the
    intensity of the input feature is higher. Features whose value is lower than
    the threshold remain silent.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of the input and output.
    :param dt: Simulation time step.
    :param sparsity: Sparsity of the input representation. 0 for no spikes and 1 for all
        spikes.
    :return: Tensor of shape ``[time, n_1, ..., n_k]``.
    """

    time = int(time / dt)

    shape = list(datum.shape)
    datum = (torch.tensor(datum) - min) / (max - min)

    # quantile = torch.quantile(datum, 1 - sparsity)
    # datum = torch.where(datum > quantile, datum, torch.zeros(shape))
    spikes = torch.zeros((time, *shape), device=device)

    timing = torch.ceil(-torch.log(datum) / tau).long()
    mask = torch.logical_and(timing >= 0, timing < time)
    spikes[timing[mask], mask] = 1

    return torch.Tensor(spikes).byte()


class SingleEncoder(Encoder):
    def __init__(self, time: int, min, max, tau=0.1, dt: float = 1.0, **kwargs):
        # language=rst
        """
        Creates a callable SingleEncoder which encodes as defined in
        ``bindsnet.encoding.single``

        :param time: Length of single spike train per input variable.
        :param dt: Simulation time step.
        :param sparsity: Sparsity of the input representation. 0 for no spikes and 1 for
            all spikes.
        """
        super().__init__(time, min=min, max=max, tau=tau, dt=dt, **kwargs)

        self.enc = single_bis


def grf(
    datum: torch.Tensor,
    time: int,
    M,
    beta,
    mean,
    std,
    var,
    density,
    min,
    max,
    dt: float = 1.0,
    device="cpu",
    **kwargs,
) -> torch.Tensor:
    # language=rst
    """
    Generates timing based single-spike encoding. Spike occurs earlier if the
    intensity of the input feature is higher. Features whose value is lower than
    the threshold remain silent.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of the input and output.
    :param dt: Simulation time step.
    :param sparsity: Sparsity of the input representation. 0 for no spikes and 1 for all
        spikes.
    :return: Tensor of shape ``[time, n_1, ..., n_k]``.
    """

    shape = list(datum.shape)
    if min > 0.0 or max != 1.0:
        datum = (torch.tensor(datum) - min) / (max - min)
    else:
        datum = torch.tensor(datum)

    r = datum.repeat((M, *([1] * (len(shape) - 1))))
    mean = mean.view(-1, *([1] * (len(shape) - 1)))

    timing = torch.round(density(r, mean, std, var) * time).squeeze().long()

    mask = torch.logical_and(timing > 0, timing < time)
    mask = torch.logical_and(mask, datum > 0)
    spikes = torch.zeros((time, *timing.shape))
    spikes[timing[mask], mask] = 1
    return torch.Tensor(spikes).byte()


class GRFEncoder(Encoder):
    def __init__(
        self,
        time: int,
        M: int = 5,
        beta: float = 1,
        dt: float = 1.0,
        min=0.0,
        max=1.0,
        **kwargs,
    ):
        time = int(time / dt)

        mean = torch.tensor([(2 * i - 3) / (2 * M - 4) for i in range(M)])
        std = 1 / (beta * np.sqrt(2 * np.pi))
        var = std**2

        density = lambda datum, mu, std, var: torch.exp(
            -((datum - mu) ** 2) / (2 * var)
        ) / (std * np.sqrt(2 * np.pi))

        super().__init__(
            time=time,
            M=M,
            beta=beta,
            mean=mean,
            std=std,
            var=var,
            min=min,
            max=max,
            density=density,
            dt=dt,
            **kwargs,
        )

        self.enc = grf
