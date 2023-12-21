# @Author: Thomas Firmin <tfirmin>
# @Date:   2023-05-19T16:04:15+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-19T18:38:03+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from sklearn.svm import SVC
import torch


def svm_fit(spikes: torch.Tensor, labels: torch.Tensor, svm: SVC) -> SVC:
    # language=rst
    """
    (Re)fit SVM model to spike data summed over time.

    :param spikes: Summed (over time) spikes of shape ``(n_examples, time, n_neurons)``.
    :param labels: Vector of shape ``(n_samples,)`` with data labels corresponding to
        spiking activity.
    :param logreg: Logistic regression model from previous fits.
    :return: (Re)fitted logistic regression model.
    """
    # (Re)fit logistic regression model.
    svm.fit(spikes, labels)
    return svm


def svm_predict(spikes: torch.Tensor, svm: SVC) -> torch.Tensor:
    # language=rst
    """
    Predicts classes according to spike data summed over time.

    :param spikes: Summed (over time) spikes of shape ``(n_examples, time, n_neurons)``.
    :param logreg: Logistic regression model from previous fits.
    :return: Predictions per example.
    """
    # Make class label predictions.
    if not hasattr(svm, "fit_status_"):
        return -1 * torch.ones(spikes.size(0)).long()

    predictions = svm.predict(spikes)
    return torch.Tensor(predictions).long()
