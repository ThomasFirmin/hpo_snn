# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-09-21T16:31:34+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-04-20T15:12:59+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)


import os
from time import time as t
import numpy as np
import torch

from typing import Iterable, List, Optional, Sequence, Tuple, Union

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
from bindsnet.network.topology import Connection

from Lie.abstract_network import AbstractNetwork


class DiehlAndCook2015(AbstractNetwork):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt,
        n_classes,
        inpt_shape,
        dt=1.0,
        reset_interval=None,
        map_size: int = 100,
        # STDP
        strength_exc: float = 22.5,
        strength_inh: float = 17.5,
        weight_decay=0.0,
        nu_pre=1e-4,
        nu_post=1e-2,
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        input_trace: float = 20.0,
        # Excit
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        e_rest=-65.0,
        e_reset=-60.0,
        e_thresh=-52.0,
        e_refrac=5,
        e_tc_decay=100.0,
        e_tc_trace=20.0,
        # Inhib
        i_rest=-60.0,
        i_reset=-45.0,
        i_thresh=-40.0,
        i_tc_decay=10.0,
        i_refrac=2,
    ) -> None:
        """
        Constructor for class ``DiehlAndCook2015``.
        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(
            n_inpt=n_inpt,
            n_classes=n_classes,
            inpt_shape=inpt_shape,
            dt=dt,
            reset_interval=reset_interval,
        )

        self.n_neurons = map_size
        self.n_out = map_size
        self.exc = strength_exc
        self.inh = strength_inh

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=input_trace
        )
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=e_rest,
            reset=e_reset,
            thresh=e_thresh,
            refrac=e_refrac,
            tc_decay=e_tc_decay,
            tc_trace=e_tc_trace,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=i_rest,
            reset=i_reset,
            thresh=i_thresh,
            tc_decay=i_tc_decay,
            refrac=i_refrac,
        )

        # Connections
        w = torch.rand(input_layer.n, exc_layer.n)

        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,
            nu=(nu_pre, nu_post),
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            weight_decay=weight_decay,
        )
        w = self.exc * torch.diag(torch.ones(exc_layer.n))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (torch.ones(inh_layer.n, exc_layer.n))
        w.fill_diagonal_(0)
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="inpt")
        self.add_layer(exc_layer, name="outpt")
        self.add_layer(inh_layer, name="inhib")
        self.add_connection(input_exc_conn, source="inpt", target="outpt")
        self.add_connection(exc_inh_conn, source="outpt", target="inhib")
        self.add_connection(inh_exc_conn, source="inhib", target="outpt")
