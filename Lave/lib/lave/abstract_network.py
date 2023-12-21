# @Author: Thomas Firmin <tfirmin>
# @Date:   2023-04-19T17:08:25+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-19T18:24:03+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
from typing import Any
import torch
import h5py
import os


class AbstractNetwork(torch.nn.Module):
    def __init__(self, n_inpt, n_classes, inpt_shape, device=None):
        super(AbstractNetwork, self).__init__()
        self.n_inpt = n_inpt
        self.n_classes = n_classes
        self.inpt_shape = inpt_shape

        self.computed_images_train = 0
        self.computed_images_test = 0

        self.layers = {}
        self.recorders_train = {"inpt": 0}
        self.recorders_test = {"inpt": 0}

        self._train_mode = True

        self.device = None

    def _update_recorders(self, spikes, layer):
        numpspikes = torch.sum(spikes)
        if self._train_mode:
            self.recorders_train[layer] += numpspikes
        else:
            self.recorders_test[layer] += numpspikes

    def forward(self, spikes):
        if self._train_mode:
            self.recorders_train["inpt"] += torch.sum(spikes).to(
                dtype=int, device="cpu"
            )
            self.computed_images_train += int(spikes.shape[0])
        else:
            self.recorders_test["inpt"] += torch.sum(spikes).to(dtype=int, device="cpu")
            self.computed_images_test += int(spikes.shape[0])

    def register_layer(self, name, layer):
        self.layers[name] = layer

    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        # network export to hdf5 format
        h = h5py.File(os.path.join(filepath, "network.pt"), "w")
        layer = h.create_group("layer")
        for i, b in enumerate(self.layers.values()):
            b.export_hdf5(layer.create_group(f"{i}"))
