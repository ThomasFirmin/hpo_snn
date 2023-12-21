# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-09-28T18:48:58+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-19T18:52:09+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


import os
import sys
from time import time as t

import numpy as np
import torch
from tqdm import tqdm

from bindsnet.encoding import (
    PoissonEncoder,
    SingleEncoder,
    BernoulliEncoder,
    RankOrderEncoder,
    NullEncoder,
)

from bindsnet.evaluation import (
    all_activity,
    assign_labels,
    proportion_weighting,
    ngram,
    update_ngram_scores,
    logreg_fit,
    logreg_predict,
)
from Lie.evaluation import svm_fit, svm_predict

from bindsnet.network.monitors import Monitor
from bindsnet.learning.learning import NoOp


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from Lie.encoders import GRFEncoder

from typing import Iterable, List, Optional, Sequence, Tuple, Union
import traceback


class Objective(object):
    def __init__(
        self,
        network,
        dataset,
        classes,
        size,
        input_features,
        input_shape,
        time,
        dataset_valid=None,
        early_stopping=None,
        split=0.80,
        dt=1.0,
        update_interval=250,
        gpu=True,
        recorders=["outpt"],
    ):
        # available decoders
        self.available_decoders = ["all", "vote", "2gram", "log", "svm"]

        # maximum number of epochs
        self.max_epoch = 3

        self.network_type = network
        # compute loss and other info every `update_interval` observations
        self.update_interval = update_interval

        self.recorders = recorders

        # Sets up Gpu use
        if gpu:
            if isinstance(gpu, str):
                self.gpu = gpu
                self.device = gpu
            else:
                self.gpu = gpu
                self.device = "cuda"
        else:
            self.gpu = gpu
            self.device = "cpu"

        self.dt = dt
        self.input_features = input_features
        self.input_shape = input_shape
        self.time = time

        # train/test split
        if isinstance(dataset, tuple):
            self.split = True
            self.train_dataset = dataset[0]
            self.test_dataset = dataset[1]
        else:
            self.split = False
            (
                self.train_dataset,
                self.test_dataset,
                _,
            ) = torch.utils.data.random_split(
                dataset,
                (
                    int(split * size),
                    size - int(split * size),
                    len(dataset) - size,
                ),
            )

        self.n_train = len(self.train_dataset)
        self.n_test = len(self.test_dataset)
        self.n_valid = 1

        # Neuron assignments
        self.n_classes = classes

        self.total_class_train = {}
        self.correct_class_train = {}
        self.accuracy_train = {}

        self.total_class_test = {}
        self.correct_class_test = {}
        self.accuracy_test = {}

        self.total_class_valid = {}
        self.correct_class_valid = {}
        self.accuracy_valid = {}

        self.accuracy_train_epoch = {}

        for key in self.available_decoders:
            self.accuracy_train[key] = []
            self.accuracy_train_epoch[key] = [None] * self.max_epoch
            self.correct_class_train[key] = [0] * self.n_classes
            self.total_class_train[key] = [0] * self.n_classes

            self.accuracy_test[key] = 0
            self.correct_class_test[key] = [0] * self.n_classes
            self.total_class_test[key] = [0] * self.n_classes

            self.accuracy_valid[key] = 0
            self.correct_class_valid[key] = [0] * self.n_classes
            self.total_class_valid[key] = [0] * self.n_classes

        self.processed_images = 0
        self.current_nimages = 0

        self.recorders_built = True

        self.train_total_spikes_out = {k: 0 for k in self.recorders}
        self.train_class_spikes_out = {k: [0] * self.n_classes for k in self.recorders}
        self.train_total_spikes_in = 0

        self.test_total_spikes_out = {k: 0 for k in self.recorders}
        self.test_class_spikes_out = {k: [0] * self.n_classes for k in self.recorders}

        self.valid_total_spikes_out = {k: 0 for k in self.recorders}
        self.valid_class_spikes_out = {k: [0] * self.n_classes for k in self.recorders}

        self.spikes_class_train = [0] * self.n_classes
        self.spikes_class_test = [0] * self.n_classes
        self.spikes_class_valid = [0] * self.n_classes

        self.test_total_spikes_in = 0
        self.valid_total_spikes_in = 0
        self.train_time = 0
        self.test_time = 0
        self.valid_time = 0
        self.total_time = 0

        # early stopping
        self.stopped = False
        self.early_stopping = early_stopping

        self.img_spikes = {k: [] for k in self.recorders}

        self.errored = False

        self.valid_dataset = dataset_valid
        self.do_valid = False

        if dataset_valid is not None:
            self.n_valid = len(self.valid_dataset)
            self.do_valid = True

    def _build_out_recorder(self, network, time, interval):
        self.spike_record = {}

        if self.recorders_built:
            self.spikes = {}

        for layer in self.recorders:
            out_shape = tuple(network.layers[layer].shape)
            tsteps = int(time / self.dt)
            # Record spikes during the simulation.
            self.spike_record[layer] = torch.zeros(
                (
                    (
                        interval,
                        tsteps,
                        *out_shape,
                    )
                ),
                device=self.device,
            )

            if self.recorders_built:
                # Set up monitors for spikes and voltages
                self.spikes[layer] = Monitor(
                    network.layers[layer],
                    state_vars=["s"],
                    time=tsteps,
                    device=self.device,
                )

                network.add_monitor(self.spikes[layer], name="%s_spikes" % layer)

        self.recorders_built = False

        return network

    def _decoder(self, inputs, network, label_tensor, train):
        preds = {}

        preds["all"] = all_activity(
            spikes=inputs,
            assignments=network.assignments_all,
            n_labels=self.n_classes,
        )

        preds["vote"] = proportion_weighting(
            spikes=inputs,
            assignments=network.assignments_vote,
            proportions=network.proportions_vote,
            n_labels=self.n_classes,
        )

        preds["2gram"] = ngram(
            spikes=inputs,
            ngram_scores=network.ngram_scores_2,
            n_labels=self.n_classes,
            n=2,
        )

        summed = inputs.sum(dim=1)
        preds["log"] = logreg_predict(spikes=summed.cpu(), logreg=network.logreg)
        preds["svm"] = svm_predict(spikes=summed.cpu(), svm=network.svm)

        if train:
            self._update_decoder(inputs, network, label_tensor, summed)

        return preds

    def _update_decoder(self, inputs, network, label_tensor, summed):
        # print("LABELS >", label_tensor)

        # Neurons assignments

        # Assign labels to excitatory layer neurons.
        (
            network.assignments_all,
            network.proportions_all,
            network.rates_all,
        ) = assign_labels(
            spikes=self.spike_record["outpt"],
            labels=label_tensor,
            n_labels=self.n_classes,
            rates=network.rates_all,
        )

        # Assign labels to excitatory layer neurons.
        (
            network.assignments_vote,
            network.proportions_vote,
            network.rates_vote,
        ) = assign_labels(
            spikes=self.spike_record["outpt"],
            labels=label_tensor,
            n_labels=self.n_classes,
            rates=network.rates_vote,
        )

        network.ngram_scores_2 = update_ngram_scores(
            spikes=inputs,
            labels=label_tensor,
            ngram_scores=network.ngram_scores_2,
            n_labels=self.n_classes,
            n=2,
        )

        # network.ngram_scores_3 = update_ngram_scores(
        #    spikes=inputs,
        #    labels=label_tensor,
        #    ngram_scores=network.ngram_scores_3,
        #    n_labels=self.n_classes,
        #    n=3,
        # )

        network.logreg = logreg_fit(
            spikes=summed.cpu(),
            labels=label_tensor.cpu(),
            logreg=network.logreg,
        )
        network.svm = svm_fit(
            spikes=summed.cpu(), labels=label_tensor.cpu(), svm=network.svm
        )

    def _update_record_layers(self, network, current_nimages, bsize):
        for layer in self.recorders:
            # Add to spikes recording.
            spikes = self.spikes[layer].get("s").transpose(0, 1)

            self.spike_record[layer][current_nimages : current_nimages + bsize] = spikes

    def _update_info(self, network, labels):
        for layer in self.recorders:
            self.train_total_spikes_out[layer] += torch.sum(self.spike_record[layer])
            # print(f"SPIKES OUT {layer}: ", self.train_total_spikes_out[layer])
            for i, l in enumerate(labels):
                img_s = torch.sum(self.spike_record[layer][i]).cpu().item()
                self.train_class_spikes_out[layer][l] += img_s
                self.img_spikes[layer].append(int(img_s))

        # Convert the array of labels into a tensor
        label_tensor = torch.tensor(labels, device=self.device)
        all_predictions = self._decoder(
            self.spike_record["outpt"][: len(label_tensor)],
            network,
            label_tensor,
            True,
        )

        for key, predictions in all_predictions.items():
            # Compute network accuracy according to available classification strategies.
            predictions = torch.tensor(predictions, device=self.device)
            verif = label_tensor == predictions
            for r_l, pred in zip(label_tensor, verif):
                self.total_class_train[key][r_l.item()] += 1
                self.correct_class_train[key][r_l.item()] += pred.item()

            self.accuracy_train[key].append(torch.sum(verif).item() / len(label_tensor))

        # print(
        #    f"\n{self.decoder}> Accuracy: %.2f (last), %.2f (average), %.2f (best)"
        #    % (
        #        self.accuracy_train[self.decoder][-1],
        #        np.mean(self.accuracy_train[self.decoder]),
        #        np.max(self.accuracy_train[self.decoder]),
        #    )
        # )

    def train(self, network):
        # time
        start = t()

        network.train(mode=True)

        rec_size = int(
            np.ceil(self.update_interval / self.batch_size) * self.batch_size
            + self.batch_size
        )

        # Train the network.
        print("\nBegin training.\n")
        for epoch in range(self.n_epochs):
            network = self._build_out_recorder(network, self.time, rec_size)
            labels = []

            if self.stopped:
                break
            else:
                if self.early_stopping is not None:
                    self.early_stopping.reset()

            print("Progress: %d / %d " % (epoch, self.n_epochs))

            # Create a dataloader to iterate and batch data
            dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=self.gpu,
            )

            for batch in dataloader:
                updated_info = False

                # Current batch size
                current_bsize = len(batch["label"])

                # Get next input sample.
                inputs = {"inpt": batch["encoded_image"].transpose(0, 1)}
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                spikes_in = torch.sum(inputs["inpt"]).cpu()
                self.train_total_spikes_in += spikes_in

                labels.extend(batch["label"])
                for i, l in enumerate(batch["label"]):
                    self.spikes_class_train[l] += torch.sum(inputs["inpt"][:, i]).cpu()

                # Run the network on the input.
                network.run(inputs=inputs, time=self.time, input_time_dim=1)

                self._update_record_layers(network, self.current_nimages, current_bsize)

                self.processed_images += current_bsize
                self.current_nimages += current_bsize

                if self.current_nimages > self.update_interval:
                    self.current_nimages = 0
                    self._update_info(network, labels)
                    labels = []
                    updated_info = True
                    if self.early_stopping and self.early_stopping(self, network):
                        self.stopped = True
                        break

                    self.img_spikes = {k: [] for k in self.recorders}

                network.reset_state_variables()  # Reset state variables.

            if not updated_info:
                self._update_info(network, labels)
                self.img_spikes = {k: [] for k in self.recorders}

            for key in self.available_decoders:
                self.accuracy_train_epoch[key][epoch] = np.mean(
                    self.accuracy_train[key]
                )
                self.accuracy_train[key] = []

            self.current_nimages = 0
            updated_info = False
            labels = []

        print("Progress: %d / %d " % (epoch + 1, self.n_epochs))
        print("Training complete.\n")

        # time
        self.train_time = t() - start
        self.total_time += self.train_time

        return network

    def test(self, network):
        # time
        start = t()

        network.train(mode=False)
        network = self._build_out_recorder(network, self.time, 1)

        # Test the network.
        print("\nBegin testing\n")

        # Create a dataloader to iterate and batch data
        dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=self.gpu,
        )

        for step, batch in enumerate(dataloader):
            # Get next input sample.
            inputs = {
                "inpt": batch["encoded_image"].view(
                    int(self.time / self.dt), 1, *self.input_shape
                )
            }
            if self.gpu:
                inputs = {k: v.cuda(self.device) for k, v in inputs.items()}

            spikes_in = torch.sum(inputs["inpt"]).cpu()
            self.test_total_spikes_in += spikes_in

            self.spikes_class_test[batch["label"]] += spikes_in

            # Run the network on the input.
            network.run(inputs=inputs, time=self.time, input_time_dim=1)
            self.processed_images += len(batch["label"])

            # Add to spikes recording.
            for layer in self.recorders:
                # Add to spikes recording.
                self.spike_record[layer][0] = (
                    self.spikes[layer].get("s").transpose(0, 1)[0]
                )
                self.test_total_spikes_out[layer] += torch.sum(self.spike_record[layer])
                for i, l in enumerate(batch["label"]):
                    self.test_class_spikes_out[layer][l] += torch.sum(
                        self.spike_record[layer][i]
                    )

            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(batch["label"], device=self.device)
            all_predictions = self._decoder(
                self.spike_record["outpt"][: len(label_tensor)],
                network,
                label_tensor,
                False,
            )

            for key, predictions in all_predictions.items():
                # Compute network accuracy according to available classification strategies.
                predictions = torch.tensor(predictions, device=self.device)
                verif = label_tensor == predictions
                for r_l, pred in zip(label_tensor, verif):
                    self.total_class_test[key][r_l.item()] += 1
                    self.correct_class_test[key][r_l.item()] += pred.item()

                self.accuracy_test[key] += torch.sum(verif).item()

            network.reset_state_variables()  # Reset state variables.

        for key in all_predictions:
            self.accuracy_test[key] /= self.n_test

        print("\nAccuracy: %.2f" % (self.accuracy_test[self.decoder]))
        print("Testing complete.\n")

        # time
        self.test_time = t() - start
        self.total_time += self.test_time

        return network

    def valid(self, network):
        # time
        start = t()

        network.train(mode=False)
        network = self._build_out_recorder(network, self.time, 1)

        # Test the network.
        print("\nBegin validation\n")

        # Create a dataloader to iterate and batch data
        dataloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=self.gpu,
        )

        for step, batch in enumerate(dataloader):
            # Get next input sample.
            inputs = {"inpt": batch["encoded_image"].transpose(0, 1)}
            if self.gpu:
                inputs = {k: v.cuda(self.device) for k, v in inputs.items()}

            spikes_in = torch.sum(inputs["inpt"]).cpu()
            self.valid_total_spikes_in += spikes_in

            self.spikes_class_valid[batch["label"]] += spikes_in

            # Run the network on the input.
            network.run(inputs=inputs, time=self.time, input_time_dim=1)
            self.processed_images += len(batch["label"])

            # Add to spikes recording.
            for layer in self.recorders:
                # Add to spikes recording.
                self.spike_record[layer][0] = (
                    self.spikes[layer].get("s").transpose(0, 1)[0]
                )
                self.valid_total_spikes_out[layer] += torch.sum(
                    self.spike_record[layer]
                )
                for i, l in enumerate(batch["label"]):
                    self.valid_class_spikes_out[layer][l] += torch.sum(
                        self.spike_record[layer][i]
                    )

            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(batch["label"], device=self.device)
            all_predictions = self._decoder(
                self.spike_record["outpt"][: len(label_tensor)],
                network,
                label_tensor,
                False,
            )

            for key, predictions in all_predictions.items():
                # Compute network accuracy according to available classification strategies.
                predictions = torch.tensor(predictions, device=self.device)
                verif = label_tensor == predictions
                for r_l, pred in zip(label_tensor, verif):
                    self.total_class_valid[key][r_l.item()] += 1
                    self.correct_class_valid[key][r_l.item()] += pred.item()

                self.accuracy_valid[key] += torch.sum(verif).item()

            network.reset_state_variables()  # Reset state variables.

        for key in all_predictions:
            self.accuracy_valid[key] /= self.n_valid

        print("\nAccuracy: %.2f" % (self.accuracy_valid[self.decoder]))
        print("Testing complete.\n")

        # time
        self.valid_time = t() - start
        self.total_time += self.valid_time

        return network

    def accuracy(self):
        if self.errored:
            d = {
                "test": 0.0,
                "train": self.accuracy_train_epoch[self.decoder][self.n_epochs - 1],
                "train_time": self.train_time,
                "test_time": self.test_time,
                "total_time": self.total_time,
                "stopped": self.stopped,
                "train_itspikes": int(self.train_total_spikes_in),
                "test_itspikes": int(self.test_total_spikes_in),
                "processed_images": int(self.processed_images),
            }
        else:
            d = {
                "test": self.accuracy_test[self.decoder],
                "train": self.accuracy_train_epoch[self.decoder][self.n_epochs - 1],
                "train_time": self.train_time,
                "test_time": self.test_time,
                "total_time": self.total_time,
                "stopped": self.stopped,
                "train_itspikes": int(self.train_total_spikes_in),
                "test_itspikes": int(self.test_total_spikes_in),
                "processed_images": int(self.processed_images),
            }

        if self.do_valid:
            d["valid"] = self.accuracy_valid[self.decoder]
            d["valid_time"] = self.valid_time
            d["test_itspikes"] = (int(self.valid_total_spikes_in),)

        for cl in range(self.n_classes):
            d[f"in_spikes_{cl}_train"] = int(self.spikes_class_train[cl])
            d[f"in_spikes_{cl}_test"] = int(self.spikes_class_test[cl])
            if self.do_valid:
                d[f"in_spikes_{cl}_valid"] = int(self.spikes_class_valid[cl])

            for key in self.available_decoders:
                d[f"{key}_class_{cl}_count_train"] = int(
                    self.total_class_train[key][cl]
                )
                d[f"{key}_class_{cl}_preds_train"] = int(
                    self.correct_class_train[key][cl]
                )
                d[f"{key}_class_{cl}_count_test"] = int(self.total_class_test[key][cl])
                d[f"{key}_class_{cl}_preds_test"] = int(
                    self.correct_class_test[key][cl]
                )
                if self.do_valid:
                    d[f"{key}_class_{cl}_count_valid"] = int(
                        self.total_class_valid[key][cl]
                    )
                    d[f"{key}_class_{cl}_preds_valid"] = int(
                        self.correct_class_valid[key][cl]
                    )

        for layer in self.recorders:
            d[f"train_{layer}_otspikes"] = int(self.train_total_spikes_out[layer])
            d[f"test_{layer}_otspikes"] = int(self.test_total_spikes_out[layer])

            if self.do_valid:
                d[f"valid_{layer}_otspikes"] = int(self.valid_total_spikes_out[layer])

            for i in range(self.n_classes):
                d[f"class{i}_train_{layer}_otspikes"] = int(
                    self.train_class_spikes_out[layer][i]
                )
                d[f"class{i}_test_{layer}_otspikes"] = int(
                    self.test_class_spikes_out[layer][i]
                )
                if self.do_valid:
                    d[f"class{i}_valid_{layer}_otspikes"] = int(
                        self.valid_class_spikes_out[layer][i]
                    )

        for key in self.available_decoders:
            d[f"{key}_test"] = self.accuracy_test[key]
            if self.do_valid:
                d[f"{key}_valid"] = self.accuracy_valid[key]

            for e in range(self.max_epoch):
                d[f"{key}_train_e{e}"] = self.accuracy_train_epoch[key][e]

        if self.early_stopping is not None:
            constraints = self.early_stopping.to_zeroinequality()
            # print(f"CONSTRAINTS:  {constraints}")
            if isinstance(constraints, float):
                d["constraint_0"] = constraints
            else:
                for i in range(len(constraints)):
                    d[f"constraint_{i}"] = constraints[i]

        return d

    def reset(self):
        torch.cuda.empty_cache()

        self.total_class_train = {}
        self.correct_class_train = {}
        self.accuracy_train = {}

        self.total_class_test = {}
        self.correct_class_test = {}
        self.accuracy_test = {}

        self.total_class_valid = {}
        self.correct_class_valid = {}
        self.accuracy_valid = {}

        self.accuracy_train_epoch = {}

        for key in self.available_decoders:
            self.accuracy_train[key] = []
            self.accuracy_train_epoch[key] = [None] * self.max_epoch
            self.correct_class_train[key] = [0] * self.n_classes
            self.total_class_train[key] = [0] * self.n_classes

            self.accuracy_test[key] = 0
            self.correct_class_test[key] = [0] * self.n_classes
            self.total_class_test[key] = [0] * self.n_classes

            self.accuracy_valid[key] = 0
            self.correct_class_valid[key] = [0] * self.n_classes
            self.total_class_valid[key] = [0] * self.n_classes

        self.processed_images = 0
        self.current_nimages = 0

        self.recorders_built = True

        self.train_total_spikes_out = {k: 0 for k in self.recorders}
        self.train_class_spikes_out = {k: [0] * self.n_classes for k in self.recorders}
        self.train_total_spikes_in = 0

        self.test_total_spikes_out = {k: 0 for k in self.recorders}
        self.test_class_spikes_out = {k: [0] * self.n_classes for k in self.recorders}

        self.valid_total_spikes_out = {k: 0 for k in self.recorders}
        self.valid_class_spikes_out = {k: [0] * self.n_classes for k in self.recorders}

        self.spikes_class_train = [0] * self.n_classes
        self.spikes_class_test = [0] * self.n_classes
        self.spikes_class_valid = [0] * self.n_classes

        self.test_total_spikes_in = 0
        self.valid_total_spikes_in = 0
        self.train_time = 0
        self.test_time = 0
        self.valid_time = 0
        self.total_time = 0

        # early stopping
        self.stopped = False
        self.errored = False

        self.img_spikes = {k: [] for k in self.recorders}

    def _create_network(self, load, **kwargs):
        if load:
            network = bindsnet.network.load(os.path.join(load, "network.pt"))
            network.assignments = torch.load(os.path.join(load, "assignments.pt"))
            network.proportions = torch.load(os.path.join(load, "proportions.pt"))
            network.rates = torch.load(os.path.join(load, "rates.pt"))
            network.ngram_scores = torch.load(os.path.join(load, "ngram.pt"))
            logreg = LogisticRegression()
            logreg.set_params(torch.load(os.path.join(load, "logreg.pt")))
            network.logreg = logreg
            svm = SVC()
            svm.set_params(torch.load(os.path.join(load, "svm.pt")))
            network.svm = svm

        else:
            network = self.network_type(
                n_inpt=self.input_features,
                n_classes=self.n_classes,
                inpt_shape=self.input_shape,
                dt=self.dt,
                **kwargs,
            )

        return network

    def __call__(self, *args, load=False, train=True, **kwargs):
        self.reset()
        print(f"Evaluating: {args}, {kwargs}")
        # epochs, encoding_window, decoder, encoder
        self.n_epochs = kwargs.pop("epochs", 1)
        assert (
            self.n_epochs <= self.max_epoch
        ), f"Too high number of epochs, {self.n_epochs}<={self.max_epoch}"
        self.batch_size = kwargs.pop("batch_size", 1)
        self.decoder = kwargs.pop("decoder", "all")

        network = self._create_network(load, **kwargs)
        network._initialize_outputs()

        if train:
            torch.cuda.empty_cache()
            try:
                print("Device:", self.device)
                network.to(self.device)
                network = self.train(network)
            except Exception as e:
                print(traceback.format_exc(), file=sys.stderr)
                print(e, file=sys.stderr)
                error = 1
            else:
                error = 0

        try:
            torch.cuda.empty_cache()
            print("Device:", self.device)
            network.to(self.device)
            network = self.test(network)
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)
            print(e, file=sys.stderr)
            error = 2
            self.errored = True
        else:
            error = 0

        if self.do_valid:
            try:
                torch.cuda.empty_cache()
                print("Device:", self.device)
                network.to(self.device)
                network = self.valid(network)
            except Exception as e:
                print(traceback.format_exc(), file=sys.stderr)
                print(e, file=sys.stderr)
                error = 4
                self.errored = True
            else:
                error = 5

        nparameters = 0
        train_nparameters = 0
        train_zeronparameters = 0
        train_onenparameters = 0
        train_95nparameters = 0
        train_05nparameters = 0

        try:
            for key, c in network.connections.items():
                nparameters += c.w.numel()
                if not isinstance(c.update_rule, NoOp):
                    train_nparameters += c.w.numel()
                    train_zeronparameters += (c.w == 1.0).sum().item()
                    train_onenparameters += (c.w == 0.0).sum().item()
                    train_95nparameters += (c.w > 0.95).sum().item()
                    train_05nparameters += (c.w < 0.05).sum().item()
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)
            print(e, file=sys.stderr)
            error = 3

        results = self.accuracy()
        results["parameters"] = nparameters
        results["trainable"] = train_nparameters
        results["trainable_zeros"] = train_zeronparameters
        results["trainable_ones"] = train_onenparameters
        results["trainable_95"] = train_95nparameters
        results["trainable_05"] = train_05nparameters
        results["error"] = error

        if self.gpu:
            mem_stat = torch.cuda.memory_stats()

            results["active_bytes.all.allocated"] = mem_stat[
                "active_bytes.all.allocated"
            ]
            results["active_bytes.all.peak"] = mem_stat["active_bytes.all.peak"]
            results["allocated_bytes.all.allocated"] = mem_stat[
                "allocated_bytes.all.allocated"
            ]
            results["allocated_bytes.all.peak"] = mem_stat["allocated_bytes.all.peak"]

        nnodes = 0
        for key, layer in network.layers.items():
            nnodes += layer.n

        results["nodes"] = nnodes

        return results
