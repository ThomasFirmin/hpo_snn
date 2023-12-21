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


from Lie.optim_diehl_and_cook import DiehlAndCook2015
from Lie.objective import Objective
from load_dataset import get_dataset
from search_spaces import get_gendh_rate_2_cnt

from zellij.core import (
    Loss,
    MockModel,
    Experiment,
    Maximizer,
    MixedSearchspace,
    Threshold,
)

from zellij.strategies import Default

import torch
import numpy as np

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
parser.add_argument("--save", type=str, default="default_base")
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

dataset_valid, _, _, _, _, _ = get_dataset(dataset_name, False)

model = Objective(
    network=DiehlAndCook2015,
    dataset=dataset,
    dataset_valid=dataset_valid,
    classes=classes,
    size=data_size,
    input_features=input_features,
    input_shape=input_shape,
    time=max_time,
    update_interval=1200,
    gpu=gpu,
    recorders=["outpt", "inhib"],
)

if mock:
    loss = Loss(
        objective=Maximizer("valid"),
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
        objective=Maximizer("valid"),
        save=True,
        record_time=record_time,
        MPI=mpi,
        kwargs_mode=True,
        threshold=1,
    )(model)

# Decision variables
values = get_gendh_rate_2_cnt()
sp = MixedSearchspace(values, loss)

if gpu:
    loss.model.device = f"cuda"  # 2 GPUs per node

stop = Threshold(loss, "calls", calls)

solutions = [
    [
        "all",
        10,
        797,
        0.000848,
        0.00888,
        356.92461,
        433.5939,
        123.387203,
        -57.661011,
        -60.800126,
        6,
        0.044037,
        2041798.636945,
        4166.794222,
        -22.81812,
        -51.113999,
        19,
        1516.155512,
    ]
]


df = Default(sp, solutions, batch=16)

exp = Experiment(df, stop, save=save_file)
exp.run()
