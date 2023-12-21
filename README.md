# Parallel hyperparameter optimization of spiking neural networks

Experiements are executed on the Jean Zay supercomputer using slurm scripts.
It can also be launched using `mpiexec` or `mpirun`.

## Installation

One can face many issues installing SNN simulators on HPC clusters. One could face conflicts between `PyTorch` versions, CUDA and custom CUDA code of Lava-DL.
We did not face critical issues running Lava-DL and Bindsnet with `Pytorch v.2.0.1`.

### Bindsnet
Bindsnet must be installed by hand following instructions found [here](https://bindsnet-docs.readthedocs.io/installation.html).

#### Locally installing network module
Locally install `Lie` containing networks and other tools concerning the SNN part using Bindsnet with :
```
pip install -e ./Lie
```

### Lava and Lava-DL
Lava-DL must be installed by hand following instructions found [here](https://lava-nc.org/lava/notebooks/in_depth/tutorial01_installing_lava.html).

Compilation of the custom CUDA code is not thread safe. One can add following lines to their scripts to avoid issues:

```python
import os

path = f"<YOUR_FOLDER>/torch_extension_mnist_{<PROCESS_RANK>}"
if not os.path.exists(path):
    os.makedirs(path)

os.environ["TORCH_EXTENSIONS_DIR"] = path
```

#### Locally installing network module
Locally install `Lave` containing networks and other tools concerning the SNN part using Lava-DL with:
```
pip install -e ./Lave
```

### Zellij

Experiments use the in development [Zellij](https://github.com/ThomasFirmin/zellij/) version.
Please intall the Zellij version from [develop_t](https://github.com/ThomasFirmin/zellij/tree/develop_t) branch.
An OPENMPI distribution is necessaary, parallelization is made using `mpi4py`.

For the version used in these experiments:
```
$ pip install -e ./zellij
```

## Running scripts

There are 4 scripts for the main experiments:

- `experiment_1`:
  - Dataset: MNIST
  - Architecture: Diehl and Cook
  - Training STDP
  - Simulator: Bindsnet
- `experiment_2`:
  - Dataset: DVS Gesture
  - Architecture: Diehl and Cook + soft distance dependent lateral inhibition
  - Training STDP
  - Simulator: Bindsnet
- `experiment_3`:
  - Dataset: MNIST
  - Architecture: CSNN
  - Training SLAYER
  - Simulator: LAVA-DL
- `experiment_4`:
  - Dataset: DVS Gesture
  - Architecture: CSNN
  - Training SLAYER
  - Simulator: LAVA-DL

Fixed files `launch_fixed_[...].py` are scipts used to retrain multiple times a unique solution.

### Options
- `--data`: size of the dataset, `default=60000`
- `--dataset`: name of the dataset, choose between `MNIST_rate_100, MNIST_rate_25, GESTURE`. Datasets are loaded using the Bindsnet loader.
- `--calls`: Number of total calls to the loss function. (Number of SNN evaluations)
- `--mpi`: `{synchronous, asynchronous, flexible}`. Use `flexible` for these experiments.
- `--gpu`: If True use GPU.
- `--record_time`: Record evaluation time for all SNNs.
- `--save`: If a path is given, results will be save there.
- `--gpu_per_node`: Deprecated. All GPUs must be isolated within their node. (One process per GPU)

### Execution with `mpirun`

```
mpiexec -machinefile <HOSTFILE> -rankfile <RANKFILE> -n 16 python3 experiment_1.py --dataset MNIST_rate_100 --mpi flexible --gpu --data 60000 --calls 1000
```

### Results

Results of all 4 experiments are in the `results` folders.
