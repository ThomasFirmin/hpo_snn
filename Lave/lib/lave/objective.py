# @Author: Thomas Firmin <tfirmin>
# @Date:   2022-09-28T18:48:58+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-19T18:52:09+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
# @Copyright: Copyright (C) 2022 Thomas Firmin


from time import time as t

import numpy as np
import torch
import sys

import lava.lib.dl.slayer as slayer


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
        weight_norm=False,
    ):
        # maximum number of epochs
        self.max_epoch = 100

        self.current_nimages_train = 0
        self.current_nimages_test = 0

        self.update_interval = update_interval

        self.network_type = network

        self.train_outpt_spike_per_class = torch.zeros(classes, dtype=int, device="cpu")
        self.test_outpt_spike_per_class = torch.zeros(classes, dtype=int, device="cpu")
        self.valid_outpt_spike_per_class = torch.zeros(classes, dtype=int, device="cpu")

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

        # Neuron assignments
        self.n_classes = classes

        self.test_total_spikes_in = 0
        self.train_time = 0
        self.test_time = 0
        self.valid_time = 0
        self.total_time = 0

        # early stopping
        self.stopped = False
        self.early_stopping = early_stopping

        self.img_spikes = {"outpt": []}

        self.valid_dataset = dataset_valid
        self.do_valid = False

        if dataset_valid is not None:
            self.n_valid = len(self.valid_dataset)
            self.do_valid = True

        self.weight_norm = weight_norm

    def accuracy(self, network, stats):
        if self.error:
            d = {
                "valid": 0.0,
                "test": 0.0,
                "train": 0.0,
                "train_loss": 0.0,
                "test_loss": 0.0,
                "train_time": 0.0,
                "test_time": 0.0,
                "total_time": 0.0,
                "stopped": True,
                "processed_images_train": 0,
                "processed_images_test": 0,
                "error": 1,
            }

            if self.do_valid:
                d["valid"] = 0

        else:
            d = {
                "test": stats.testing.accuracy_log[-1],
                "train": stats.training.accuracy_log[-1],
                "train_loss": stats.training.loss,
                "test_loss": stats.testing.loss,
                "train_time": self.train_time,
                "test_time": self.test_time,
                "total_time": self.total_time,
                "stopped": self.stopped,
                "processed_images_train": int(network.computed_images_train),
                "processed_images_test": int(network.computed_images_test),
                "error": 0,
            }

            if self.do_valid:
                d["valid"] = stats.validation.accuracy

        if d["test"] is None:
            d["test"] = 0.0

        for layer in network.recorders_train:
            d[f"train_{layer}_otspikes"] = int(network.recorders_train[layer])
            d[f"test_{layer}_otspikes"] = int(network.recorders_test[layer])

        for e in range(self.max_epoch):
            if e < len(stats.training.accuracy_log):
                d[f"train_accuracy_e{e}"] = stats.training.accuracy_log[e]
            else:
                d[f"train_accuracy_e{e}"] = None

            if e < len(stats.testing.accuracy_log):
                d[f"test_accuracy_e{e}"] = stats.testing.accuracy_log[e]
            else:
                d[f"test_accuracy_e{e}"] = None

        for c in range(self.n_classes):
            d[f"train_ot_spikes_class{c}"] = (
                self.train_outpt_spike_per_class[c].cpu().item()
            )
            d[f"test_ot_spikes_class{c}"] = (
                self.test_outpt_spike_per_class[c].cpu().item()
            )
            if self.do_valid:
                d[f"valid_ot_spikes_class{c}"] = (
                    self.valid_outpt_spike_per_class[c].cpu().item()
                )

        if self.early_stopping:
            constraints = self.early_stopping.to_zeroinequality()
            # print(f"CONSTRAINTS:  {constraints}")
            if isinstance(constraints, float):
                if self.error:
                    d["constraint_0"] = 100
                else:
                    d["constraint_0"] = constraints
            else:
                for i in range(len(constraints)):
                    if self.error:
                        d[f"constraint_{i}"] = 100
                    else:
                        d[f"constraint_{i}"] = constraints[i]

        return d

    def reset(self):
        self.processed_images = 0
        self.test_total_spikes_in = 0
        self.train_time = 0
        self.test_time = 0
        self.valid_time = 0
        self.total_time = 0

        self.train_outpt_spike_per_class = torch.zeros(
            self.n_classes, dtype=int, device="cpu"
        )
        self.test_outpt_spike_per_class = torch.zeros(
            self.n_classes, dtype=int, device="cpu"
        )
        self.valid_outpt_spike_per_class = torch.zeros(
            self.n_classes, dtype=int, device="cpu"
        )

        # early stopping
        self.stopped = False
        self.error = False

        self.img_spikes = {"outpt": []}

        if self.early_stopping:
            self.early_stopping.reset()

    def __call__(self, *args, **kwargs):
        torch.cuda.empty_cache()
        print(f"Evaluating: {args}, {kwargs}")

        self.reset()

        self.n_epochs = kwargs.pop("epochs", 1)
        assert (
            self.n_epochs <= self.max_epoch
        ), f"Too high number of epochs, {self.n_epochs}<={self.max_epoch}"
        self.batch_size = kwargs.pop("batch_size", 16)
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.decoder = kwargs.pop("decoder", "rate")

        if self.decoder == "rate":
            error = slayer.loss.SpikeRate(true_rate=0.2, false_rate=0.03)
        else:
            error = slayer.loss.SpikeMax()

        stats = slayer.utils.LearningStats()

        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=int(self.batch_size),
            shuffle=True,
            pin_memory=self.gpu,
        )
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=int(self.batch_size),
            shuffle=True,
            pin_memory=self.gpu,
        )

        print(f"TRAINING on {self.device}")
        self.train_time = 0
        self.test_time = 0

        network = self.network_type(
            n_inpt=self.input_features,
            n_classes=self.n_classes,
            inpt_shape=self.input_shape,
            weight_norm=self.weight_norm,
            **kwargs,
        )

        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
        assistant = slayer.utils.Assistant(
            network,
            error,
            optimizer,
            stats,
            classifier=slayer.classifier.Rate.predict,
        )

        torch.cuda.empty_cache()
        # Network to GPU
        if self.gpu:
            print("Device:", self.device)
            network.to(self.device)

        try:
            print("STARTING TRAINING PHASE")
            for epoch in range(self.n_epochs):
                if self.early_stopping:
                    self.early_stopping.reset()
                network._train_mode = True

                start = t()
                for i, batch in enumerate(train_dataloader):  # training loop
                    input = (
                        batch["encoded_image"]
                        .permute(0, 2, 3, 4, 1)
                        .type(torch.float32)
                        .to(self.device)
                    )
                    label = batch["label"]
                    self.current_nimages_train += len(label)
                    output = assistant.train(input, label.to(self.device))

                    dims = (1, 2, 3, 4)
                    ssum = torch.sum(output, dim=dims).to(dtype=int, device="cpu")

                    self.train_outpt_spike_per_class[label.cpu()] += ssum
                    self.img_spikes["outpt"].extend(list(ssum))

                    if self.current_nimages_train > self.update_interval:
                        self.current_nimages_train = 0
                        stats_str = str(stats).replace("| ", "\n")
                        print(
                            f"""
                            [Epoch {epoch:2d}/{self.n_epochs}: batch {i}/{len(train_dataloader)}]\n
                            {stats_str}\n
                            Computed images: {network.computed_images_train}\n
                            Spikes train: {network.recorders_train}\n
                            Spikes test: {network.recorders_test}\n
                            IMG SPIKES: {len(self.img_spikes['outpt'])}\n
                            """
                        )
                        if self.early_stopping and self.early_stopping(self, network):
                            self.stopped = True
                            break

                        self.img_spikes = {"outpt": []}

                self.train_time += t() - start
                self.current_nimages_train = 0
                self.img_spikes = {"outpt": []}
                network._train_mode = False

                start = t()
                for i, batch in enumerate(test_dataloader):  # training loop
                    input = (
                        batch["encoded_image"]
                        .permute(0, 2, 3, 4, 1)
                        .type(torch.float32)
                        .to(self.device)
                    )
                    label = batch["label"]
                    output = assistant.test(input, label.to(self.device))

                    dims = (1, 2, 3, 4)
                    ssum = torch.sum(output, dim=dims).to(dtype=int, device="cpu")
                    self.test_outpt_spike_per_class[label.cpu()] += ssum

                self.test_time += t() - start
                stats.update()
                if self.stopped:
                    break

            print(f"END TRAINING on {self.device}")
        except Exception as e:
            print("ERROR OCCURED")
            print(e, file=sys.stderr)
            self.error = True

        if self.do_valid:
            print("DOING VALIDATION")
            # Network to GPU
            if self.gpu:
                print("Device:", self.device)
                network.to(self.device)

            valid_dataloader = torch.utils.data.DataLoader(
                self.valid_dataset,
                batch_size=int(self.batch_size),
                shuffle=True,
                pin_memory=self.gpu,
            )
            network._train_mode = False
            start = t()
            for i, batch in enumerate(valid_dataloader):  # training loop
                input = (
                    batch["encoded_image"]
                    .permute(0, 2, 3, 4, 1)
                    .type(torch.float32)
                    .to(self.device)
                )
                label = batch["label"]
                output = assistant.valid(input, label.to(self.device))

                dims = (1, 2, 3, 4)
                ssum = torch.sum(output, dim=dims).to(dtype=int, device="cpu")
                self.valid_outpt_spike_per_class[label.cpu()] += ssum

                print(f"""[VALIDATION: batch {i}/{len(valid_dataloader)}]\n""")

            self.valid_time += t() - start
            print("ENDING VALIDATION")

        results = self.accuracy(network, stats)
        nparameters = 0
        train_nparameters = 0
        train_zeronparameters = 0
        train_onenparameters = 0
        train_95nparameters = 0
        train_05nparameters = 0

        try:
            for p in network.parameters():
                numel = p.numel()
                nparameters += numel
                if p.requires_grad:
                    train_nparameters += numel
                    train_zeronparameters += (
                        (p == 1.0).sum().to(dtype=int, device="cpu").item()
                    )
                    train_onenparameters += (
                        (p == 0.0).sum().to(dtype=int, device="cpu").item()
                    )
                    train_95nparameters += (
                        (p > 0.95).sum().to(dtype=int, device="cpu").item()
                    )
                    train_05nparameters += (
                        (p < 0.05).sum().to(dtype=int, device="cpu").item()
                    )
        except:
            pass

        results["parameters"] = nparameters
        results["trainable"] = train_nparameters
        results["trainable_zeros"] = train_zeronparameters
        results["trainable_ones"] = train_onenparameters
        results["trainable_95"] = train_95nparameters
        results["trainable_05"] = train_05nparameters

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

        print(f"RETURNING")
        return results
