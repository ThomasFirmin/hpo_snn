# @Author: Thomas Firmin <tfirmin>
# @Date:   2023-04-20T12:35:18+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-12T14:35:42+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from bindsnet.datasets import create_torchvision_dataset_wrapper
from Lie.encoders import GRFEncoder
from bindsnet.encoding import (
    PoissonEncoder,
    SingleEncoder,
    BernoulliEncoder,
    RankOrderEncoder,
    NullEncoder,
)

import os


def get_dataset(dataset_name, train=True):
    if dataset_name == "MNIST_rate_25":
        from bindsnet.datasets import MNIST
        from torchvision import transforms

        dataset = MNIST(
            PoissonEncoder(time=25, dt=1.0),
            None,
            root=os.path.join("data", "mnist"),
            download=True,
            train=train,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * 512)]
            ),
        )
        print(len(dataset))
        encode = True
        input_features = 784
        input_shape = (1, 28, 28)
        classes = 10
        time_step = 25

    elif dataset_name == "MNIST_rate_50":
        from bindsnet.datasets import MNIST
        from torchvision import transforms

        dataset = MNIST(
            PoissonEncoder(time=50, dt=1.0),
            None,
            root=os.path.join("data", "mnist"),
            download=True,
            train=train,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * 256)]
            ),
        )
        print(len(dataset))
        encode = True
        input_features = 784
        input_shape = (1, 28, 28)
        classes = 10
        time_step = 50

    elif dataset_name == "MNIST_rate_100":
        from bindsnet.datasets import MNIST
        from torchvision import transforms

        dataset = MNIST(
            PoissonEncoder(time=100, dt=1.0),
            None,
            root=os.path.join("data", "mnist"),
            download=True,
            train=train,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * 128)]
            ),
        )
        print(len(dataset))
        encode = True
        input_features = 784
        input_shape = (1, 28, 28)
        classes = 10
        time_step = 100

    elif dataset_name == "MNIST_rate_250":
        from bindsnet.datasets import MNIST
        from torchvision import transforms

        dataset = MNIST(
            PoissonEncoder(time=250, dt=1.0),
            None,
            root=os.path.join("data", "mnist"),
            download=True,
            train=train,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * 128)]
            ),
        )
        print(len(dataset))
        encode = True
        input_features = 784
        input_shape = (1, 28, 28)
        classes = 10
        time_step = 250

    elif dataset_name == "MNIST_latency":
        from bindsnet.datasets import MNIST
        from torchvision import transforms

        dataset = MNIST(
            GRFEncoder(M=5, beta=1.5, time=100, dt=1.0),
            None,
            root=os.path.join("data", "mnist"),
            download=True,
            train=train,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
        print(len(dataset))
        encode = True
        input_features = 3920
        input_shape = (5, 28, 28)
        classes = 10
        time_step = 100

    elif dataset_name == "NMNIST":
        import tonic

        transform = tonic.transforms.Compose(
            [
                tonic.transforms.Denoise(filter_time=10000),
                tonic.transforms.ToFrame(sensor_size=(34, 34, 2), n_time_bins=100),
                tonic.transforms.NumpyAsType(bool),
                tonic.transforms.NumpyAsType(int),
            ]
        )
        nmnist = create_torchvision_dataset_wrapper(tonic.datasets.NMNIST)
        dataset = nmnist(
            save_to=os.path.join("data", "nmnist"),
            train=True,
            transform=transform,
        )
        print(len(dataset))
        encode = False
        input_features = 2312
        input_shape = (2, 34, 34)
        classes = 10
        time_step = 100
    elif dataset_name == "GESTURE":
        import tonic

        transform = tonic.transforms.Compose(
            [
                tonic.transforms.Denoise(filter_time=5000),
                tonic.transforms.ToFrame(sensor_size=(128, 128, 2), n_time_bins=100),
                tonic.transforms.NumpyAsType(bool),
                tonic.transforms.NumpyAsType(int),
            ]
        )
        gesture = create_torchvision_dataset_wrapper(tonic.datasets.DVSGesture)
        dataset = gesture(
            save_to=os.path.join("data", "gesture"),
            train=train,
            transform=transform,
        )
        print(len(dataset))
        encode = False
        input_features = 32768
        input_shape = (2, 128, 128)
        classes = 11
        time_step = 100
    elif dataset_name == "GESTURE_50":
        import tonic

        transform = tonic.transforms.Compose(
            [
                tonic.transforms.Denoise(filter_time=5000),
                tonic.transforms.ToFrame(sensor_size=(128, 128, 2), n_time_bins=100),
                tonic.transforms.NumpyAsType(bool),
                tonic.transforms.NumpyAsType(int),
            ]
        )
        gesture = create_torchvision_dataset_wrapper(tonic.datasets.DVSGesture)
        dataset = gesture(
            save_to=os.path.join("data", "gesture"),
            train=train,
            transform=transform,
        )
        print(len(dataset))
        encode = False
        input_features = 32768
        input_shape = (2, 128, 128)
        classes = 11
        time_step = 100
    else:
        raise ValueError(f"Unknown dataset name, got {dataset_name}")

    return dataset, encode, input_features, input_shape, classes, time_step
