# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-10-03T10:18:51+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-19T18:03:04+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-10-03T10:18:51+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-19T18:03:04+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


from Lie.optim_diehl_and_cook_dvs import DiehlAndCook2015
from Lie.objective import Objective
from Lie.early_stoping import NonSpikingPercentageStopping
from load_dataset import get_dataset
from search_spaces import get_gendh_rate_2_cnt_dvs
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
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=int, default=60000)
parser.add_argument("--calls", type=int, default=100000)
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--mpi", type=str, default=False)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--mock", dest="mock", action="store_true")
parser.add_argument("--record_time", dest="record_time", action="store_true")
parser.add_argument("--save", type=str, default="scbo_stdp_dvs_corrected")
parser.add_argument("--gpu_per_node", type=int, default=1)

parser.set_defaults(gpu=False, verbose=True)

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

model = Objective(
    network=DiehlAndCook2015,
    dataset=dataset,
    classes=classes,
    size=data_size,
    input_features=input_features,
    input_shape=input_shape,
    time=max_time,
    update_interval=10,
    gpu=gpu,
    recorders=["outpt", "inhib"],
    early_stopping=NonSpikingPercentageStopping("outpt", 1, 0.3, 862)
    | NonSpikingPercentageStopping("inhib", 1, 0.3, 862),
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
                "test": lambda *arg, **kwarg: np.random.random(),
                "train": lambda *arg, **kwarg: np.random.random(),
            }
        )
    )
else:
    loss = Loss(
        objective=Maximizer("test"),
        constraint=["constraint_0", "constraint_1"],
        save=True,
        record_time=record_time,
        MPI=mpi,
        kwargs_mode=True,
        threshold=1,
    )(model)

# Decision variables
values = get_gendh_rate_2_cnt_dvs()
sp = ContinuousSearchspace(values, loss, converter=Basic())


turbo_state = CTurboState(
    sp.size,
    best_constraint_values=(
        torch.ones(
            2,
        )
        * torch.inf
    ),
    batch_size=16,
)


if gpu:
    loss.model.device = f"cuda"  # 2 GPUs per node

stop = Threshold(loss, "calls", calls)

covar_module = ScaleKernel(
    MaternKernel(
        nu=2.5, ard_num_dims=sp.size, lengthscale_constraint=Interval(0.005, 4.0)
    )
)
noise_constraint = Interval(1e-8, 1e-3)

bo = SCBO(
    sp,
    turbo_state,
    batch_size=16,
    initial_size=50,
    gpu=True,
    covar_module=covar_module,
    noise_constraint=noise_constraint,
    cholesky_size=500,
    beam=10000,
)


exp = Experiment(bo, stop, save=save_file)
exp.run()
