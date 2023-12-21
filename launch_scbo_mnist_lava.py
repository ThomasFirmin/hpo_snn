# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-10-03T10:18:51+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-16T19:08:43+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


from lave.optim_conv import ConvMNIST
from lave.objective import Objective
from lave.early_stoping import NonSpikingPercentageStopping
from load_dataset import get_dataset
from search_spaces import get_mnist_rate_lava_cnt
from zellij.utils.converters import Basic

from zellij.core import (
    Loss,
    MockModel,
    Experiment,
    Maximizer,
    ContinuousSearchspace,
    Threshold,
)

from zellij.strategies import SCBO, CTurboState

import torch
import numpy as np

from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.models.approximate_gp import SingleTaskVariationalGP
from gpytorch.mlls import VariationalELBO

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=int, default=60000)
parser.add_argument("--calls", type=int, default=100000)
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--mpi", type=str, default=False)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--mock", dest="mock", action="store_true")
parser.add_argument("--record_time", dest="record_time", action="store_true")
parser.add_argument("--save", type=str, default="scbo_lava_mnist")
parser.add_argument("--gpu_per_node", type=int, default=1)

parser.set_defaults(gpu=True, record_time=True)

args = parser.parse_args()
data_size = args.data
dataset_name = args.dataset
calls = args.calls
mpi = args.mpi
gpu = args.gpu
mock = args.mock
record_time = args.record_time
save_file = args.save
gpu_per_node = args.gpu_per_node

dataset, encode, input_features, input_shape, classes, max_time = get_dataset(
    dataset_name
)


import os

model = Objective(
    network=ConvMNIST,
    dataset=dataset,
    classes=classes,
    size=data_size,
    input_features=input_features,
    input_shape=input_shape,
    time=max_time,
    update_interval=1200,
    gpu=gpu,
    early_stopping=NonSpikingPercentageStopping("outpt", 5, 0.05, 48000),
)

if mock:
    loss = Loss(
        objective=Maximizer("test"),
        save=True,
        record_time=record_time,
        MPI=mpi,
        kwargs_mode=True,
    )(
        MockModel(
            {
                "valid": lambda *arg, **kwarg: np.random.random(),
                "train": lambda *arg, **kwarg: np.random.random(),
            }
        )
    )
else:
    loss = Loss(
        objective=Maximizer("test"),
        constraint=["constraint_0"],
        save=True,
        record_time=record_time,
        MPI=mpi,
        kwargs_mode=True,
        threshold=1,
    )(model)

# Decision variables
values = get_mnist_rate_lava_cnt()
sp = ContinuousSearchspace(values, loss, converter=Basic())


turbo_state = CTurboState(
    sp.size,
    best_constraint_values=(
        torch.ones(
            1,
        )
        * torch.inf
    ),
    batch_size=32,
)

if gpu:
    loss.model.device = f"cuda"  # 2 GPUs per node

path = f"/gpfswork/rech/vmz/ujq94yi/torch_extension_mnist_{loss.rank}"
if not os.path.exists(path):
    os.makedirs(path)

os.environ["TORCH_EXTENSIONS_DIR"] = path

stop = Threshold(loss, "calls", calls)

covar_module = ScaleKernel(
    MaternKernel(
        nu=2.5, ard_num_dims=sp.size, lengthscale_constraint=Interval(0.005, 4.0)
    )
)
noise_constraint = Interval(1e-8, 1e-3)

if loss.rank != 0:
    bo = SCBO(
        sp,
        turbo_state,
        batch_size=32,
        initial_size=1000,
        gpu=False,
        covar_module=covar_module,
        noise_constraint=noise_constraint,
        cholesky_size=500,
        beam=100000,
    )
else:
    bo = SCBO(
        sp,
        turbo_state,
        batch_size=32,
        initial_size=1000,
        gpu=True,
        covar_module=covar_module,
        noise_constraint=noise_constraint,
        cholesky_size=500,
        beam=100000,
    )


exp = Experiment(bo, stop, save=save_file)
exp.run()
