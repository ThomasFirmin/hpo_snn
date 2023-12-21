from zellij.core.metaheuristic import ContinuousMetaheuristic
from zellij.strategies.tools.turbo_state import update_c_state

import torch
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

from botorch.models import SingleTaskGP
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models.transforms.outcome import Standardize
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll

import os

import logging

logger = logging.getLogger("zellij.scbo")


class SCBO(ContinuousMetaheuristic):
    """Scalable Constrained Bayesian Optimization

    Works in the unit hypercube. :code:`converter` :ref:`addons` are required.

    See `SCBO <https://botorch.org/tutorials/scalable_constrained_bo>`_.
    It is based on `BoTorch <https://botorch.org/>`_ and `GPyTorch <https://gpytorch.ai/>`__.

    Attributes
    ----------
    search_space : Searchspace
        Search space object containing bounds of the search space
    turbo_state : CTurboState
        :code:`CTurboState` object.
    verbose : bool
        If False, there will be no print.
    surrogate : botorch.models.model.Model, default=SingleTaskGP
        Gaussian Process Regressor object from 'botorch'.
        Determines the surrogate model that Bayesian optimization will use to
        interpolate the loss function
    mll : gpytorch.mlls, default=ExactMarginalLogLikelihood
            Object from gpytorch.mlls it determines which marginal loglikelihood to use
            when optimizing kernel's hyperparameters
    likelihood : gpytorch.likelihoods, default=GaussianLikelihood
        Object from gpytorch.likelihoods defining the likelihood.
    batch_size : int, default=4
        Number of solutions sampled within the surrogate, to return at each iteration.
    n_canditates : int, default=None
        Number of candidates to sample with the surrogate.
    initial_size : int, default=10
        Size of the initial set of solution to draw randomly.
    cholesky_size : int, default=800
        Maximum size for which Lanczos method is used instead of Cholesky decomposition.
    beam : int, default=2000
        Maximum number of solutions that can be stored and used to compute the Gaussian Process.
    gpu: bool, default=True
        Use GPU if available
    kwargs
        Key word arguments linked to the surrogate, mll or likelihood.

    See Also
    --------
    :ref:`meta` : Parent class defining what a Metaheuristic is
    :ref:`lf` : Describes what a loss function is in Zellij
    :ref:`sp` : Describes what a loss function is in Zellij
    """

    def __init__(
        self,
        search_space,
        turbo_state,
        verbose=True,
        surrogate=SingleTaskGP,
        mll=ExactMarginalLogLikelihood,
        likelihood=GaussianLikelihood,
        batch_size=4,
        n_candidates=None,
        initial_size=10,
        cholesky_size=800,
        beam=2000,
        gpu=False,
        **kwargs,
    ):
        """__init__

        Parameters
        ----------
        search_space : Searchspace
            Search space object containing bounds of the search space
        turbo_state : CTurboState
            :code:`CTurboState` object.
        verbose : bool
            If False, there will be no print.
        surrogate : botorch.models.model.Model, default=SingleTaskGP
            Gaussian Process Regressor object from 'botorch'.
            Determines the surrogate model that Bayesian optimization will use to
            interpolate the loss function
        mll : gpytorch.mlls, default=ExactMarginalLogLikelihood
                Object from gpytorch.mlls it determines which marginal loglikelihood to use
                when optimizing kernel's hyperparameters
        likelihood : gpytorch.likelihoods, default=GaussianLikelihood
            Object from gpytorch.likelihoods defining the likelihood.
        batch_size : int, default=4
            Number of solutions sampled within the surrogate, to return at each iteration.
        n_canditates : int, default=None
            Number of candidates to sample with the surrogate.
        initial_size : int, default=10
            Size of the initial set of solution to draw randomly.
        cholesky_size : int, default=800
            Maximum size for which Lanczos method is used instead of Cholesky decomposition.
        beam : int, default=2000
            Maximum number of solutions that can be stored and used to compute the Gaussian Process.
        gpu: bool, default=True
            Use GPU if available
        kwargs
            Key word arguments linked to the surrogate, mll or likelihood.
        """

        super().__init__(search_space, verbose)

        assert (
            search_space.loss.constraint is not None
        ), "Loss function must have constraints. `loss.constraint` is None"

        ##############
        # PARAMETERS #
        ##############

        self.surrogate = surrogate
        self.mll = mll
        self.likelihood = likelihood

        self.batch_size = batch_size
        self.n_candidates = n_candidates
        self.initial_size = initial_size

        self.beam = beam

        self.kwargs = kwargs

        #############
        # VARIABLES #
        #############
        self.nconstraint = len(self.search_space.loss.constraint)

        self.turbo_state = turbo_state

        # Determine if BO is initialized or not
        self.initialized = False

        # Number of iterations
        self.iterations = 0

        if gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        self.dtype = torch.double

        # Prior points
        self.train_x = torch.empty(
            (0, self.search_space.size), dtype=self.dtype, device=self.device
        )
        # Prior objective
        self.train_obj = torch.empty((0, 1), dtype=self.dtype, device=self.device)
        # Prior constraints
        self.train_c = torch.empty(
            (0, self.nconstraint),
            dtype=self.dtype,
            device=self.device,
        )

        self.sobol = SobolEngine(dimension=self.search_space.size, scramble=True)

        self._build_kwargs()

        self.cmodels_list = [None] * self.nconstraint

        # Count generated models
        self.models_number = 0

        self.cholesky_size = cholesky_size

        self.iterations = 0

    def _build_kwargs(self):
        # Surrogate kwargs
        self.model_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.surrogate.__init__.__code__.co_varnames
        }

        for m in self.model_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        # Likelihood kwargs
        self.likelihood_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.likelihood.__init__.__code__.co_varnames
        }
        for m in self.likelihood_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        # MLL kwargs
        self.mll_kwargs = {
            key: value
            for key, value in self.kwargs.items()
            if key in self.mll.__init__.__code__.co_varnames
        }
        for m in self.mll_kwargs.values():
            if isinstance(m, torch.nn.Module):
                m.to(self.device)

        logger.debug(self.model_kwargs, self.likelihood_kwargs, self.mll_kwargs)

    def _generate_initial_data(self):
        return self.search_space.random_point(self.initial_size)

    # Initialize a surrogate
    def _initialize_model(self, train_x, train_obj, state_dict=None):
        train_x.to(self.device, dtype=self.dtype)
        train_obj.to(self.device, dtype=self.dtype)

        likelihood = self.likelihood(**self.likelihood_kwargs)

        # define models for objective and constraint
        model = self.surrogate(
            train_x,
            train_obj,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
            **self.model_kwargs,
        )
        model.to(self.device)

        if "num_data" in self.mll.__init__.__code__.co_varnames:
            mll = self.mll(
                model.likelihood,
                model.model,
                num_data=train_x.shape[-2],  # type: ignore
                **self.mll_kwargs,
            )
        else:
            mll = self.mll(
                model.likelihood,
                model,
                **self.mll_kwargs,
            )

        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return mll, model, train_obj

    def generate_batch(
        self,
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        batch_size,
        n_candidates,  # Number of candidates for Thompson sampling
        constraint_model,
    ):
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()

        # Add weights based trust region
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        ################################

        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=self.dtype, device=self.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, dtype=self.dtype, device=self.device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
            model=model, constraint_model=constraint_model, replacement=False
        )
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

        return X_next.detach()

    def reset(self):
        """reset()

        reset :code:`Bayesian_optimization` to its initial state.

        """
        self.initialized = False
        self.train_x = torch.empty(
            (0, self.search_space.size), dtype=self.dtype, device=self.device
        )
        self.train_y = torch.empty((0, 1), dtype=self.dtype, device=self.device)
        self.train_c = torch.empty(
            (0, len(self.search_space.loss.contraint)),
            dtype=self.dtype,
            device=self.device,
        )

    def forward(self, X, Y, constraint):
        """forward

        Runs one step of BO.

        Parameters
        ----------
        X : list
            List of previously computed points
        Y : list
            List of loss value linked to :code:`X`.
            :code:`X` and :code:`Y` must have the same length.

        Returns
        -------
        points
            Return a list of new points to be computed with the :ref:`lf`.
        info
            Additionnal information linked to :code:`points`

        """

        if self.turbo_state.restart_triggered:
            self.initialized = False
            self.turbo_state.reset()

        if not self.initialized:
            # call helper functions to generate initial training data and initialize model
            train_x = self._generate_initial_data()
            self.initialized = True
            return train_x, {
                "iteration": self.iterations,
                "algorithm": "InitSCBO",
                "length": 1.0,
                "trestart": self.turbo_state.restart_triggered,
                "model": -1,
            }
        else:
            torch.cuda.empty_cache()
            if (X is not None and Y is not None) and (len(X) > 0 and len(Y) > 0):
                self.iterations += 1

                new_x = torch.tensor(X, dtype=self.dtype, device=self.device)
                new_obj = -torch.tensor(
                    Y, dtype=self.dtype, device=self.device
                ).unsqueeze(-1)
                new_c = torch.tensor(constraint, dtype=self.dtype, device=self.device)

                # update training points
                self.train_x = torch.cat([self.train_x, new_x], dim=0)
                self.train_obj = torch.cat([self.train_obj, new_obj], dim=0)
                self.train_c = torch.cat([self.train_c, new_c], dim=0)

                # Remove worst solutions from the beam
                if len(self.train_x) > self.beam:
                    sidx = torch.argsort(self.train_obj)

                    self.train_x = self.train_x[sidx]
                    self.train_obj = self.train_obj[sidx]
                    self.train_c = self.train_c[sidx]

                    violation = self.train_c.sum(dim=1)
                    nvidx = violation < 0

                    new_x = self.train_x[nvidx][: self.beam]
                    new_obj = self.train_obj[nvidx][: self.beam]
                    new_c = self.train_c[nvidx][: self.beam]

                    if len(new_x) < self.beam:
                        nfill = self.beam - len(new_x)
                        violeted = torch.logical_not(nvidx)
                        v_x = self.train_x[violeted]
                        v_obj = self.train_obj[violeted]
                        v_c = self.train_c[violeted]

                        scidx = torch.argsort(v_c)[nfill]

                        # update training points
                        self.train_x = torch.cat([new_x, v_x[scidx]], dim=0)
                        self.train_obj = torch.cat([new_obj, v_obj[scidx]], dim=0)
                        self.train_c = torch.cat([new_c, v_c[scidx]], dim=0)
                    else:
                        self.train_x = new_x
                        self.train_obj = new_obj
                        self.train_c = new_c

                # If initial size not reached, returns 1 additionnal solution
                if len(self.train_obj) < self.initial_size:
                    return [self.search_space.random_point(1)], {
                        "iteration": self.iterations,
                        "algorithm": "AddInitSCBO",
                        "length": 1.0,
                        "trestart": self.turbo_state.restart_triggered,
                        "model": -1,
                    }
                else:
                    self.turbo_state = update_c_state(
                        state=self.turbo_state, Y_next=new_obj, C_next=new_c
                    )
                    with gpytorch.settings.max_cholesky_size(self.cholesky_size):
                        # reinitialize the models so they are ready for fitting on next iteration
                        # use the current state dict to speed up fitting
                        mll, model, train_Y = self._initialize_model(
                            self.train_x,
                            self.train_obj,
                            state_dict=None,
                        )

                        for i in range(self.nconstraint):
                            cmll, cmodel, _ = self._initialize_model(
                                self.train_x,
                                self.train_c[:, i].unsqueeze(-1),
                                state_dict=None,
                            )
                            try:
                                fit_gpytorch_mll(cmll)
                                self.cmodels_list[i] = cmodel  # type: ignore

                            except ModelFittingError:
                                print(
                                    f"In SCBO, ModelFittingError for constraint: {self.search_space.loss.constraint[i]}, previous fitted model will be used."
                                )

                        try:
                            fit_gpytorch_mll(mll)
                        except ModelFittingError:
                            return self.search_space.random_point(len(Y)), {
                                "iteration": self.iterations,
                                "algorithm": "FailedSCBO",
                                "length": 1.0,
                                "trestart": self.turbo_state.restart_triggered,
                                "model": -1,
                            }

                        # optimize and get new observation
                        new_x = self.generate_batch(
                            state=self.turbo_state,
                            model=model,
                            X=self.train_x,
                            Y=train_Y,
                            batch_size=self.batch_size,
                            n_candidates=self.n_candidates,
                            constraint_model=ModelListGP(*self.cmodels_list),  # type: ignore
                        )

                        self.save(model, self.cmodels_list)

                        return new_x.cpu().numpy(), {
                            "iteration": self.iterations,
                            "algorithm": "SCBO",
                            "length": self.turbo_state.length,
                            "trestart": self.turbo_state.restart_triggered,
                            "model": self.models_number,
                        }
            else:
                return [self.search_space.random_point(1)], {
                    "iteration": self.iterations,
                    "algorithm": "ResampleSCBO",
                    "length": 1.0,
                    "trestart": self.turbo_state.restart_triggered,
                    "model": -1,
                }

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["cmodels_list"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cmodels_list = [None] * len(self.search_space.loss.constraint)

    def save(self, model, cmodels):
        path = self.search_space.loss.folder_name
        foldername = os.path.join(path, "scbo")
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        std_dict = model.state_dict()
        std_dict["nlengthscale"] = model.covar_module.base_kernel.lengthscale

        torch.save(
            std_dict,
            os.path.join(foldername, f"{self.models_number}_model.pth"),
        )
        for idx, m in enumerate(cmodels):
            std_dict = m.state_dict()
            std_dict["nlengthscale"] = m.covar_module.base_kernel.lengthscale
            torch.save(
                std_dict,
                os.path.join(foldername, f"{self.models_number}_c{idx}model.pth"),
            )
        self.models_number += 1
