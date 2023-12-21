# @Author: Thomas Firmin <tfirmin>
# @Date:   2023-04-19T17:08:25+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-19T18:24:03+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)
from Lie.network import Network
from functools import reduce
from operator import mul
import torch
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class AbstractNetwork(Network):
    def __init__(self, n_inpt, n_classes, inpt_shape, dt=1.0, reset_interval=None):
        super().__init__(dt=dt, reset_interval=reset_interval)
        self.n_inpt = n_inpt
        self.n_classes = n_classes
        self.inpt_shape = inpt_shape

    def _initialize_outputs(self):
        output_size = reduce(mul, self.layers["outpt"].shape)
        self.assignments_all = -torch.ones(output_size)
        self.proportions_all = torch.zeros((output_size, self.n_classes))
        self.rates_all = torch.zeros((output_size, self.n_classes))

        self.assignments_vote = -torch.ones(output_size)
        self.proportions_vote = torch.zeros((output_size, self.n_classes))
        self.rates_vote = torch.zeros((output_size, self.n_classes))

        self.ngram_scores_2 = {}
        self.logreg = LogisticRegression()
        self.svm = SVC()

    def to(self, device=None, *args, **kwargs):
        self.assignments_all = self.assignments_all.to(device=device)
        self.proportions_all = self.proportions_all.to(device=device)
        self.rates_all = self.rates_all.to(device=device)

        self.assignments_vote = self.assignments_vote.to(device=device)
        self.proportions_vote = self.proportions_vote.to(device=device)
        self.rates_vote = self.rates_vote.to(device=device)

        self.ngram_scores_2 = {
            k: v.to(device=device) for k, v in self.ngram_scores_2.items()
        }

        return super().to(device=device, *args, **kwargs)

    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        super().save(os.path.join(filepath, "network.pt"))

        # Max spikes
        torch.save(
            self.assignments_all,
            open(os.path.join(filepath, "assignments_all.pt"), "wb"),
        )
        torch.save(
            self.proportions_all,
            open(os.path.join(filepath, "proportion_all.pt"), "wb"),
        )
        torch.save(self.rates_all, open(os.path.join(filepath, "rates_all.pt"), "wb"))

        # Avg spikes
        torch.save(
            self.assignments_vote,
            open(os.path.join(filepath, "assignments_vote.pt"), "wb"),
        )
        torch.save(
            self.proportions_vote,
            open(os.path.join(filepath, "proportion_vote.pt"), "wb"),
        )
        torch.save(self.rates_vote, open(os.path.join(filepath, "rates_vote.pt"), "wb"))

        # n-rgam
        torch.save(self.ngram_scores_2, open(os.path.join(filepath, "2gram.pt"), "wb"))

        torch.save(
            self.logreg.get_params(),
            open(os.path.join(filepath, "logreg.pt"), "wb"),
        )

        torch.save(self.svm.get_params(), open(os.path.join(filepath, "svm.pt"), "wb"))
