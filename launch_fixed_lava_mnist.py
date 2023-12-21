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


from lave.optim_conv import ConvMNIST
from lave.objective import Objective
from lave.early_stoping import NonSpikingPercentageStopping
from load_dataset import get_dataset
from search_spaces import get_dvs_rate_lava_cnt
from zellij.utils.converters import Basic

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
parser.add_argument("--data", type=int, default=60000)
parser.add_argument("--calls", type=int, default=100000)
parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--mpi", type=str, default=False)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--mock", dest="mock", action="store_true")
parser.add_argument("--record_time", dest="record_time", action="store_true")
parser.add_argument("--save", type=str, default="fixed_lava_mnist")
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
dataset_valid, _, _, _, _, _ = get_dataset(dataset_name, False)

import os

model = Objective(
    network=ConvMNIST,
    dataset=dataset,
    classes=classes,
    size=data_size,
    input_features=input_features,
    input_shape=input_shape,
    time=max_time,
    dataset_valid=dataset_valid,
    update_interval=1200,
    gpu=gpu,
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
        save=True,
        record_time=record_time,
        MPI=mpi,
        kwargs_mode=True,
        threshold=1,
    )(model)

# Decision variables
values = get_dvs_rate_lava_cnt()
sp = MixedSearchspace(values, loss)

if gpu:
    loss.model.device = f"cuda"  # 2 GPUs per node

path = f"/gpfswork/rech/vmz/ujq94yi/torch_extension_mnist_{loss.rank}"
if not os.path.exists(path):
    os.makedirs(path)

os.environ["TORCH_EXTENSIONS_DIR"] = path

print(f"TORCH_EXTENSIONS_DIR : {path}")

stop = Threshold(loss, "calls", calls)

solutions = [
    [
        100,
        35,
        "rate",
        123,
        46,
        12,
        11,
        0.0020529112952568,
        0.108554592362159,
        0.6668261310513325,
        1.3353865579330946,
        0.1089225086730582,
        0.5603692958603232,
        0.1353143859023598,
        0.256212411776361,
        0.1160740150590444,
        0.029057478601254,
    ]
]


df = Default(sp, solutions, batch=16)

exp = Experiment(df, stop, save=save_file)
exp.run()
