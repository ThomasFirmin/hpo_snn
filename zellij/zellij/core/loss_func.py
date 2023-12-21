# @Author: Thomas Firmin <ThomasFirmin>
# @Date:   2022-05-03T15:41:48+02:00
# @Email:  thomas.firmin@univ-lille.fr
# @Project: Zellij
# @Last modified by:   tfirmin
# @Last modified time: 2023-05-23T13:19:21+02:00
# @License: CeCILL-C (http://www.cecill.info/index.fr.html)

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import os
import time

from zellij.core.objective import Minimizer
import logging
from queue import Queue
from datetime import datetime

logger = logging.getLogger("zellij.loss")

try:
    from mpi4py import MPI
except ImportError as err:
    logger.info(
        "To use MPILoss object you need to install mpi4py and an MPI distribution\n\
    You can use: pip install zellij[MPI]"
    )


class LossFunc(ABC):

    """LossFunc

    LossFunc allows to wrap function of type :math:`f(x)=y`.
    With :math:`x` a set of hyperparameters.
    However, **Zellij** supports alternative pattern:
    :math:`f(x)=results,model` for example.
    Where:

        * :math:`results` can be a `list <https://docs.python.org/3/tutorial/datastructures.html#more-on-lists>`__ or a `dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`__. Be default the first element of the list or the dictionary is considered as the loss vale.
        * :math:`model` is optionnal, it is an object with a save() method. (e.g. a neural network from Tensorflow)

    You must wrap your function so it can be used in Zellij by adding
    several features, such as calls count, saves, parallelization...

    Attributes
    ----------
    model : function
        Function of type :math:`f(x)=y` or :math:`f(x)=results,model. :math:`x`
        must be a solution. A solution can be a list of float, int...
        It can also be of mixed types...
    objective : Objective, default=Minimizer
        Objectve object determines what and and how to optimize.
        (minimization, maximization, ratio...)
    best_score : float
        Best score found so far.
    best_point : list
        Best solution found so far.
    calls : int
        Number of loss function calls

    See Also
    --------
    Loss : Wrapper function
    MPILoss : Distributed version of LossFunc
    SerialLoss : Basic version of LossFunc
    """

    def __init__(
        self,
        model,
        objective=Minimizer,
        save: Optional[str] = None,
        record_time=False,
        only_score=False,
        kwargs_mode=False,
        default=None,
        constraint=None,
    ):
        """__init__(model, save=False)

        Parameters
        ----------
        model : Callable
            Function of type :math:`f(x)=y`. With :math:`x` a solution, a set
            of hyperparameters for example.
            And :math:`y` can be a single value, a list, a tuple, or a dict,
            containing the loss value and other optionnal information.
            It can also be of mixed types, containing, strings, float, int...
        objective : Objective, default=Minimizer
            An :code:`Objective` object determines what the optimization problem is.
            If :code:`objective` is :code:`Maximizer`, then the first argument
            of the object, list, tuple or dict, returned by the :code:`__call__`
            function will be maximized.
        save : str, optionnal
            If a :code:`str` is given, then outputs will be saved in :code:`save`.
        record_time : boolean, default=False
            If True, :code:`start_time`, :code:`end_time`, :code:`start_date`, :code:`end_date` will be recorded
            and saved in the save file for each :code:`__call__`.
        only_score : bool, default=False
            If True, then only the score of evaluated solutions are saved.
            Otherwise, all infos returned by the :ref:`lf` and :ref:`meta` are
            saved.
        kwargs_mode : bool, default=False
            If True, then solutions are passed as kwargs to :ref:`lf`. Keys, are
            the names of the :ref:`var` within the :ref:`sp`.
        default : dict, optionnal
            Dictionnary of defaults arguments, kwargs, to pass to the loss function.
            They are not affected by any :ref:`metaheuristic` or other methods.
        constraint : list[str], default=None
            Constraints works when the model returns a dictionnary of values.
            Constraints values returned by the model must be booleans.
            If a list of strings is passed, constraints values will be passed to
            the :code:`forward` method of :ref:`meta`.

        """
        ##############
        # PARAMETERS #
        ##############

        self.model = model
        self.objective = objective

        self.save = save
        self.record_time = record_time

        self.only_score = only_score
        self.kwargs_mode = kwargs_mode

        self.default = default
        self.constraint = constraint

        #############
        # VARIABLES #
        #############

        self.best_score = float("inf")
        self.best_point = None

        self.calls = 0

        self.labels = []

        self.outputs_path = ""
        self.model_path = ""
        self.plots_path = ""
        self.loss_file = ""

        self.file_created = False

        self._init_time = time.time()

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        if isinstance(value, type):
            self._objective = value()
        else:
            self._objective = value

    @property
    def save(self):
        return self._save

    @save.setter
    def save(self, value):
        self._save = value
        if isinstance(value, str):
            self.folder_name = value
        else:
            self.folder_name = f"{self.model.__class__.__name__}_zlj_save"

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["model"]
        del state["_init_time"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        logger.warning("In Loss, after unpickling, the `model` has to be set manually.")
        self.model = None
        self._init_time = time.time()

    @abstractmethod
    def _save_model(self, *args):
        """_save_model()

        Private abstract method to save a model.
        Be carefull, to be exploitable, the initial loss func must be of form
        :math:`f(x) = (y, model)`, :math:`y` are the results of the evaluation of :math:`x`
        by :math:`f`. :math:`model` is optional, if you want to save the best model
        found (e.g. a neural network) you can return the model.
        However the model must have a "save" method with a filename.
        (e.g. model.save(filename)).

        """
        pass

    @abstractmethod
    def __call__(self, X, stop_obj=None, **kwargs):
        pass

    def _compute_loss(self, point):
        if self.kwargs_mode:
            new_kwargs = {key: value for key, value in zip(self.labels, point)}  # type: ignore
            if self.default:
                new_kwargs.update(self.default)

            start = time.time()
            start_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

            # compute loss
            res, trained_model = self._build_return(self.model(**new_kwargs))  # type: ignore

            end = time.time()
            end_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

            if self.record_time:
                res["eval_time"] = end - start
                res["start_time"] = start - self._init_time
                res["end_time"] = end - self._init_time
                res["start_date"] = start_date
                res["end_date"] = end_date
        else:
            start = time.time()
            start_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

            if self.default:
                lossv = self.model(point, **self.default)  # type: ignore
            else:
                lossv = self.model(point)  # type: ignore

            res, trained_model = self._build_return(lossv)
            end = time.time()
            end_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

            if self.record_time:
                res["eval_time"] = end - start
                res["start_time"] = start - self._init_time
                res["end_time"] = end - self._init_time
                res["start_date"] = start_date
                res["end_date"] = end_date

        return res, trained_model

    def _create_file(self, x, *args):
        """create_file(x, *args)

        Create a save file:

        Structure:

            foldername
            | model # if sav = True in LossFunc, contains model save
              | model_save
            | outputs # Contains loss function outputs
              | file_1.csv
              | ...
            | plots # if save = True while doing .show(), contains plots
              | plot_1.png
              | ...

        Parameters
        ----------
        solution : list
            Needs a solution to determine the header of the save file

        *args : list[label]
            Additionnal info to add after the score/evaluation of a point.

        """

        # Create a valid folder
        try:
            os.makedirs(self.folder_name)
        except FileExistsError:
            pass

        # Create ouputs folder
        self.outputs_path = os.path.join(self.folder_name, "outputs")
        # Create a valid folder
        try:
            os.makedirs(self.outputs_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"Outputs folder already exists, got {self.outputs_path}. The experiment will end. Try another folder to save your experiment."
            )

        self.model_path = os.path.join(self.folder_name, "model")
        # Create a valid folder
        try:
            os.makedirs(self.model_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"Model folder already exists, got {self.model_path}. The experiment will end. Try another folder to save your experiment."
            )

        self.plots_path = os.path.join(self.folder_name, "plots")
        # Create a valid folder
        try:
            os.makedirs(self.plots_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"Plots folder already exists, got {self.plots_path}. The experiment will end. Try another folder to save your experiment."
            )

        # Additionnal header for the outputs file
        if len(args) > 0:
            suffix = "," + ",".join(str(e) for e in args)
        else:
            suffix = ""

        # Create base outputs file for loss func
        self.loss_file = os.path.join(self.outputs_path, "all_evaluations.csv")

        # Determine header
        if len(self.labels) != len(x):
            logger.warning(
                "WARNING: Labels are of incorrect size, it will be replaced in the save file header"
            )
            for i in range(len(x)):
                self.labels.append(f"attribute{i}")

        with open(self.loss_file, "w") as f:
            if self.only_score:
                f.write("objective\n")
            else:
                f.write(",".join(str(e) for e in self.labels) + suffix + "\n")

        logger.info(
            f"INFO: Results will be saved at: {os.path.abspath(self.folder_name)}"
        )

        self.file_created = True

    def _save_file(self, x, **kwargs):
        """_save_file(x, **kwargs)

        Private method to save information about an evaluation of the loss function.

        Parameters
        ----------
        x : list
            Solution to save.
        **kwargs : dict, optional
            Other information to save linked to x.
        """

        if not self.file_created:
            self._create_file(x, *list(kwargs.keys()))

        # Determine if additionnal contents must be added to the save
        if len(kwargs) > 0:
            suffix = ",".join(str(e) for e in kwargs.values())
        else:
            suffix = ""

        # Save a solution and additionnal contents
        with open(self.loss_file, "a+") as f:
            if self.only_score:
                f.write(f"{kwargs['objective']}\n")
            else:
                f.write(",".join(str(e) for e in x) + "," + suffix + "\n")

    # Save best found solution
    def _save_best(self, x, y):
        """_save_best(x, y)

        Save point :code:`x` with score :code:`y`, and verify if this point is the best found so far.

        Parameters
        ----------
        x : list
            Set of hyperparameters (a solution)
        y : {float, int}
            Loss value (score) associated to x.

        """

        self.calls += 1

        # Save best
        if y < self.best_score:
            self.best_score = y
            self.best_point = list(x)[:]

    def _build_return(self, r):
        """_build_return(r)

        This method builds a unique return according to the outputs of the loss function

        Parameters
        ----------
        r : {list, float, int}
            Returns of the loss function

        Returns
        -------
        rd : dict
            Dictionnary mapping outputs from the loss function

        model : object
            Model object with a 'save' method

        """

        # Separate results and model
        if isinstance(r, tuple):
            if len(r) > 1:
                results, model = r
            else:
                results, model = r, False
        else:
            results, model = r, False

        return self.objective(results), model

    def reset(self):
        """reset()

        Reset all attributes of :code:`LossFunc` at their initial values.

        """

        self.best_score = float("inf")
        self.best_point = None
        self.best_argmin = None

        self.calls = 0

        self.labels = []

        self.outputs_path = ""
        self.model_path = ""
        self.plots_path = ""
        self.loss_file = ""

        self.file_created = False

        self._init_time = time.time()


class SerialLoss(LossFunc):

    """SerialLoss

    SerialLoss adds methods to save and evaluate the original loss function.

    Methods
    -------

    __call__(X, filename='', **kwargs)
        Evaluate a list X of solutions with the original loss function.

    _save_model(score, source)
        See LossFunc, save a model according to its score and the worker rank.

    See Also
    --------
    Loss : Wrapper function
    LossFunc : Inherited class
    MPILoss : Distributed version of LossFunc
    """

    def __init__(
        self,
        model,
        objective=Minimizer,
        save: Optional[str] = None,
        record_time=False,
        only_score=False,
        kwargs_mode=False,
        default=None,
        constraint=None,
        **kwargs,
    ):
        """__init__(model, save=False, verbose=True)

        Initialize SerialLoss.

        """

        super().__init__(
            model,
            objective,
            save,
            record_time,
            only_score,
            kwargs_mode,
            default,
            constraint,
        )

    def __call__(self, X, stop_obj=None, **kwargs):
        """__call__(model, **kwargs)

        Evaluate a list X of solutions with the original loss function.

        Parameters
        ----------
        X : list
            List of solutions to evaluate. be carefull if a solution is a list X must be a list of lists.
        **kwargs : dict, optional
            Additionnal informations to save before the score.

        Returns
        -------
        res : list
            Return a list of all the scores corresponding to each evaluated solution of X.

        """
        res = [None] * len(X)
        if self.constraint is None:
            list_constraint = None
        else:
            list_constraint = np.ones((len(X), len(self.constraint)), dtype=float)

        for idx, x in enumerate(X):
            outputs, model = self._compute_loss(x)

            res[idx] = outputs["objective"]
            if self.constraint:
                list_constraint[idx] = [outputs[k] for k in self.constraint]  # type: ignore

            # Saving
            if self.save:
                # Save model into a file if it is better than the best found one
                if model:
                    self._save_model(outputs["objective"], model)

                # Save score and solution into a file
                self._save_file(x, **outputs, **kwargs)

            self._save_best(x, outputs["objective"])

        return X, res, list_constraint

    def _save_model(self, score, trained_model):
        # Save model into a file if it is better than the best found one
        if score < self.best_score:
            save_path = os.path.join(
                self.model_path, f"{self.model.__class__.__name__}_best"
            )
            if hasattr(trained_model, "save") and callable(
                getattr(trained_model, "save")
            ):
                os.system(f"rm -rf {save_path}")
                trained_model.save(save_path)
            else:
                logger.error("Model/loss function does not have a method called `save`")
                exit()


class MPILoss(LossFunc):
    def __init__(
        self,
        model,
        objective=Minimizer,
        save: Optional[str] = None,
        record_time=False,
        only_score=False,
        kwargs_mode=False,
        strategy="synchronous",
        workers=None,
        default=None,
        constraint=None,
        **kwargs,
    ):
        """MPILoss

        MPILoss adds method to dynamically distribute the evaluation
        of multiple solutions within a distributed environment, where a version of
        `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`__
        is available.

        Attributes
        ----------
        model : Callable
            Function of type :math:`f(x)=y`. With :math:`x` a solution, a set
            of hyperparameters for example.
            And :math:`y` can be a single value, a list, a tuple, or a dict,
            containing the loss value and other optionnal information.
            It can also be of mixed types, containing, strings, float, int...
        objective : Objective, default=Minimizer
            An :code:`Objective` object determines what the optimization problem is.
            If :code:`objective` is :code:`Maximizer`, then the first argument
            of the object, list, tuple or dict, returned by the :code:`__call__`
            function will be maximized.
        save : str, optionnal
            If a :code:`str` is given, then outputs will be saved in :code:`save`.
        verbose : bool, default=False
            Verbosity of the loss function.
        only_score : bool, default=False
            If True, then only the score of evaluated solutions are saved.
            Otherwise, all infos returned by the :ref:`lf` and :ref:`meta` are
            saved.
        kwargs_mode : bool, default=False
            If True, then solutions are passed as kwargs to :ref:`lf`. Keys, are
            the names of the :ref:`var` within the :ref:`sp`.
        strategy : str, default=synchronous
            if :code:`strategy='synchronous`: then :code:`__call__` will return all results from all
            solutions passed, once all of them have been evaluated.
            if :code:`strategy='asynchronous`: then :code:`__call__` will return
            the result from an evaluation of a solution assoon as it receives a
            result from a worker. Other solutions, are still being evaluated in
            background.
            if :code:`strategy='flexible`: then :code:`__call__` will return
            all computed results, only if the number of remaining uncomputed solutions
            is below a certain threshold. Pass: :code:`threshold=int` kwarg, to :code:`Loss`
            or :code:`MPILoss`.
        workers : int, optionnal
            Number of workers among the total number of processes spawned by
            MPI. At least, one process is dedicated to the master.
        comm : MPI_COMM_WORLD
            All created processes and their communication context are grouped in comm.
        status : MPI_Status
            Data structure containing information about a received message.
        rank : int
            Process rank
        p : int
            comm size
        master : boolean
            If True the process is the master, else it is the worker.

        See Also
        --------
        Loss : Wrapper function
        LossFunc : Inherited class
        SerialLoss : Basic version of LossFunc
        """
        #################
        # MPI VARIABLES #
        #################

        try:
            self.comm = MPI.COMM_WORLD
            self.status = MPI.Status()
            self.p_name = MPI.Get_processor_name()

            self.rank = self.comm.Get_rank()
            self.p = self.comm.Get_size()
        except Exception as err:
            logger.error(
                """To use MPILoss object you need to install mpi4py and an MPI
                distribution.\nYou can use: pip install zellij[Parallel]"""
            )

            raise err

        ###############
        # PAARAMETERS #
        ###############

        super().__init__(
            model,
            objective,
            save,
            record_time,
            only_score,
            kwargs_mode,
            default,
            constraint,
        )

        if workers:
            self.workers_size = workers
        else:
            self.workers_size = self.p - 1

        self.recv_msg = 0
        self.sent_msg = 0

        self._personnal_folder = os.path.join(
            os.path.join(self.folder_name, "tmp_wks"), f"worker{self.rank}"
        )

        self.strat_name = strategy

        # Strategy kwargs
        self.skwargs = kwargs

        self.pqueue = Queue()

        # list of idle workers
        self.idle = list(range(1, self.workers_size + 1))

        # loss worker rank : (point, id, info, source)
        self.p_historic = {
            i: [None, None, None, None] for i in range(1, self.workers_size + 1)
        }  # historic of points sent to workers

        self._strategy = None  # set to None for definition issue

        self.is_master = self.rank == 0
        self.is_worker = self.rank != 0

        # Property, defines parallelisation strategy
        self._master_rank = 0  # type: ignore

    @property
    def save(self):
        return self._save

    @save.setter
    def save(self, value):
        self._save = value
        if isinstance(value, str):
            self.folder_name = value
            self._personnal_folder = os.path.join(
                os.path.join(self.folder_name, "tmp_wks"), f"worker{self.rank}"
            )
        else:
            self.folder_name = f"{self.model.__class__.__name__}_zlj_save"
            self._personnal_folder = os.path.join(
                os.path.join(self.folder_name, "tmp_wks"), f"worker{self.rank}"
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["comm"]
        del state["status"]
        del state["p_name"]
        del state["rank"]
        del state["p"]
        del state["_personnal_folder"]
        del state["pqueue"]
        del state["idle"]
        del state["p_historic"]
        del state["_strategy"]
        del state["is_master"]
        del state["is_worker"]
        del state["_MPILoss__master_rank"]
        del state["model"]
        del state["_init_time"]
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.update(state)

        try:
            self.comm = MPI.COMM_WORLD
            self.status = MPI.Status()
            self.p_name = MPI.Get_processor_name()

            self.rank = self.comm.Get_rank()
            self.p = self.comm.Get_size()
        except Exception as err:
            logger.error(
                """To use MPILoss object you need to install mpi4py and an MPI
                distribution.\nYou can use: pip install zellij[Parallel]"""
            )

            raise err

        self._personnal_folder = os.path.join("tmp_wks", f"worker{self.rank}")

        self.pqueue = Queue()

        # list of idle workers
        self.idle = list(range(1, self.workers_size + 1))

        # loss worker rank : (point, id, info, source)
        self.p_historic = {
            i: [None, None, None, None] for i in range(1, self.workers_size + 1)
        }  # historic of points sent to workers

        self._strategy = None  # set to None for definition issue

        self.is_master = self.rank == 0
        self.is_worker = self.rank != 0

        # Property, defines parallelisation strategy
        self._master_rank = 0  # type: ignore

    @property
    def _master_rank(self):
        return self.__master_rank

    @_master_rank.setter
    def _master_rank(self, value):
        self.__master_rank = value

        if self.rank == value:
            self.is_master = True

        if self.strat_name == "asynchronous":
            self._strategy = _MonoAsynchronous_strat(self, value, **self.skwargs)
        elif self.strat_name == "synchronous":
            self._strategy = _MonoSynchronous_strat(self, value, **self.skwargs)
        elif self.strat_name == "flexible":
            self._strategy = _MonoFlexible_strat(self, value, **self.skwargs)
        else:
            raise NotImplementedError(
                f"""
                    {self.strat_name} parallelisation is not implemented.
                    Use MPI='asynchronous', 'synchronous', 'flexible', or False for non
                    distributed loss function.
                    """
            )

    def __call__(self, X, stop_obj=None, **kwargs):
        new_x, score, constraints = self._strategy(X, stop_obj=stop_obj, **kwargs)  # type: ignore

        return new_x, score, constraints

    def master(self, pqueue=None, stop_obj=None):
        """master()

        Evaluate a list :code:`X` of solutions with the original loss function.

        Returns
        -------
        res : list
            Return a list of all the scores corresponding to each evaluated solution of X.

        """

        logger.debug(f"Master of rank {self.rank} Starting")

        # if there is a stopping criterion
        if stop_obj:
            stopping = stop_obj
        else:
            stopping = lambda *args: False
        # second stopping criterion determine by the parallelization itself
        cnt = True

        if pqueue:
            self.pqueue = pqueue

        while not stopping() and cnt:
            # Send solutions to workers
            # if a worker is idle and if there are solutions
            while not self.comm.iprobe() and (
                len(self.idle) > 0 and not self.pqueue.empty()
            ):
                self._send_point(self.idle, self.pqueue, self.p_historic)

            if self.comm.iprobe():
                msg = self.comm.recv(status=self.status)
                cnt = self._parse_message(
                    msg, self.pqueue, self.p_historic, self.idle, self.status
                )

        logger.debug(f"MASTER{self.rank}, calls:{self.calls} |!| STOPPING |!|")

        if stopping():
            logger.debug(f"MASTER{self.rank}, sending STOP")
            self._stop()

    def _parse_message(self, msg, pqueue, historic, idle, status):
        tag = status.Get_tag()
        source = status.Get_source()

        # receive score
        if tag == 1:
            (
                point,
                outputs,
                point_id,
                point_info,
                point_source,
            ) = self._recv_score(msg, source, idle, historic)

            # Save score and solution into the object
            self._save_best(point, outputs["objective"])
            if self.save:
                # Save model into a file if it is better than the best found one
                self._save_model(outputs["objective"], source)

                # Save score and solution into a file
                self._save_file(point, **outputs, **point_info)

            cnt = self._process_outputs(
                point, outputs, point_id, point_info, point_source
            )
        # receive a point to add to the queue
        elif tag == 2:
            cnt = self._recv_point(msg, source, pqueue)
            # STOP
        elif tag == 9:
            cnt = False
        # error: abort
        else:
            logger.error(
                f"Unknown message tag, got {tag} from process {source}. Processes will abort"
            )
            cnt = False

        return cnt

    # send point from master to worker
    def _send_point(self, idle, pqueue, historic):
        next_point = pqueue.get()
        dest = idle.pop()
        historic[dest] = next_point
        logger.debug(
            f"MASTER {self.rank} sending point to WORKER {dest}.\n Remaining points in queue: {pqueue.qsize()}"
        )
        self.comm.send(dest=dest, tag=0, obj=next_point[0])

    # receive a new point to put in the point queue. (from a forward)
    def _recv_point(self, msg, source, pqueue):
        logger.debug(
            f"MASTER {self.rank} receiving point from PROCESS {source}\n{msg}\n"
        )
        pqueue.put(msg)
        return True

    # receive score from workers
    def _recv_score(self, msg, source, idle, historic):
        logger.debug(
            f"MASTER {self.rank} receiving score from WORKER {source} : {msg}, historic : {historic[source]}"
        )
        point = historic[source][0][:]
        point_id = historic[source][1]
        point_info = historic[source][2].copy()
        point_source = historic[source][3]
        historic[source] = [None, None, None, None]

        outputs = msg

        idle.append(source)

        return point, outputs, point_id, point_info, point_source

    def _process_outputs(self, point, outputs, id, info, source):
        return self._strategy._process_outputs(point, outputs, id, info, source)  # type: ignore

    def _stop(self):
        """stop()

        Send a stop message to all processes.

        """
        logger.debug(f"MASTER {self.rank} sending stop message")
        for i in range(0, self.p):
            if i != self.rank:
                self.comm.send(dest=i, tag=9, obj=False)

    def _save_model(self, score, source):
        """_save_model(score, source)

        Be carefull, to be exploitable, the initial loss func must be of form
        :math:`f(x) = (y, model)`, :math:`y` are the results of the evaluation of :math:`x`
        by :math:`f`. :math:`model` is optional, if you want to save the best model
        found (e.g. a neural network) you can return the model.
        However the model must have a "save" method with a filename.
        (e.g. model.save(filename)).

        Parameters
        ----------

        score : int
            Score corresponding to the solution saved by the worker.
        source : int
            Worker rank which evaluate a solution and return score

        """

        # Save model into a file if it is better than the best found one
        if score < self.best_score:
            master_path = os.path.join(self.model_path, f"{self.folder_name}_best")
            worker_path = os.path.join(
                os.path.join(self.folder_name, "tmp_wks"), f"worker{source}"
            )

            if os.path.isdir(worker_path):
                os.system(f"rm -rf {master_path}")
                os.system(f"cp -rf {worker_path} {master_path}")

    def _wsave_model(self, model):
        if hasattr(model, "save") and callable(getattr(model, "save")):
            os.system(f"rm -rf {self._personnal_folder}")
            model.save(self._personnal_folder)
        else:
            logger.error("The model does not have a method called save")

    def worker(self):
        """worker()

        Initialize worker. While it does not receive a stop message,
        a worker will wait for a solution to evaluate.

        """

        logger.debug(f"WORKER {self.rank} starting")

        stop = True

        while stop:
            logger.debug(f"WORKER {self.rank} receving message")
            # receive message from master
            msg = self.comm.recv(source=self._master_rank, status=self.status)  # type: ignore
            tag = self.status.Get_tag()
            source = self.status.Get_source()

            if tag == 9:
                logger.debug(f"WORKER{self.rank} |!| STOPPING |!|")
                stop = False

            elif tag == 0:
                logger.debug(f"WORKER {self.rank} receved a point, {msg}")
                point = msg
                # Verify if a model is returned or not
                outputs, model = self._compute_loss(point)

                # Save the model using its save method
                if model and self.save:
                    logger.debug(f"WORKER {self.rank} saving model")
                    if model:
                        self._wsave_model(model)

                # Send results
                logger.debug(
                    f"WORKER {self.rank} sending {outputs} to {self._master_rank}"
                )
                self.comm.send(dest=self._master_rank, tag=1, obj=outputs)  # type: ignore

            else:
                logger.debug(f"WORKER {self.rank} unknown tag, got {tag}")


class _Parallel_strat:
    def __init__(self, loss, master_rank, **kwargs):
        super().__init__()
        self.master_rank = master_rank
        try:
            self.comm = MPI.COMM_WORLD
        except Exception as err:
            logger.error(
                """To use MPILoss object you need to install mpi4py and an MPI
                distribution.\nYou can use: pip install zellij[Parallel]"""
            )

            raise err

        # counter for computed point
        self._computed = 0

        self._lf = loss


# Mono Synchrone -> Save score return list of score
class _MonoSynchronous_strat(_Parallel_strat):
    # Executed by Experiment to compute X
    def __call__(self, X, stop_obj=None, **kwargs):
        pqueue = Queue()
        for id, x in enumerate(X):
            pqueue.put((x, id, kwargs, None))

        self.y = [None] * len(X)
        if self._lf.constraint is None:
            self.return_constraint = False
            self.list_constraint = None
        else:
            self.return_constraint = True
            self.list_constraint = np.ones(
                (len(X), len(self._lf.constraint)), dtype=float
            )

        self._lf.master(pqueue, stop_obj=stop_obj)

        return X, self.y[:], self.list_constraint

    # Executed by master when it receives a score from a worker
    # Here Meta master and loss master are the same process, so shared memory
    def _process_outputs(self, point, outputs, id, info, source):
        self.y[id] = outputs["objective"]

        if self.return_constraint:
            self.list_constraint[id] = [outputs[k] for k in self._lf.constraint]  # type: ignore

        self._computed += 1

        if self._computed < len(self.y):
            logger.info(
                f"COMPUTED POINT {self._computed}/{len(self.y)}, calls:{self._lf.calls}"
            )
            return True
        else:
            logger.info(
                f"STOP COMPUTED POINT {self._computed}/{len(self.y)}, calls:{self._lf.calls}"
            )
            self._computed = 0
            return False


class _MonoAsynchronous_strat(_Parallel_strat):
    def __init__(self, loss, master_rank, **kwargs):
        super().__init__(loss, master_rank, **kwargs)
        self._current_points = {}
        self._computed_point = (None, None)

    # send a point to loss master
    def _send_to_master(self, point, **kwargs):
        id = self._computed  # number of computed points used as id
        self._current_points[id] = point
        self._lf.pqueue.put((point, id, kwargs, None))
        self._computed += 1

    # Executed by Experiment to compute X
    def __call__(self, X, stop_obj=None, **kwargs):
        # send point, point ID and point info
        for point in X:
            self._send_to_master(point, **kwargs)  # send point

        self._lf.master(stop_obj=stop_obj)

        new_x, outputs = (
            self._computed_point[0].copy(),  # type: ignore
            self._computed_point[1].copy(),  # type: ignore
        )
        self._computed_point = (None, None)

        if self._lf.constraint is None:
            return [new_x], [outputs["objective"]], None
        else:
            return (
                [new_x],
                [outputs["objective"]],
                [[outputs[k] for k in self._lf.constraint]],
            )

    # Executed by master when it receives a score from a worker
    def _process_outputs(self, point, outputs, id, info, source):
        self._computed_point = (point, outputs)
        return False


class _MonoFlexible_strat(_Parallel_strat):
    def __init__(self, loss, master_rank, threshold):
        super().__init__(loss, master_rank)
        self._current_points = {}

        self._flex_x = []
        self._flex_y = []
        self._flex_c = []

        # When queue size is < threshold, the loss master stop, so meta can return new points
        self.threshold = threshold

    # send a point to loss master
    def _send_to_master(self, point, **kwargs):
        id = self._computed  # number of computed points used as id
        self._current_points[id] = point
        self._lf.pqueue.put((point, id, kwargs, None))
        self._computed += 1

    # Executed by Experiment to compute X
    def __call__(self, X, stop_obj=None, **kwargs):
        self._flex_x = []
        self._flex_y = []
        self._flex_c = []

        # send point, point ID and point info
        for point in X:
            self._send_to_master(point, **kwargs)  # send point

        self._lf.master(stop_obj=stop_obj)

        if self._lf.constraint is None:
            return self._flex_x, self._flex_y, None
        else:
            return self._flex_x, self._flex_y, self._flex_c

    # Executed by master when it receives a score from a worker
    def _process_outputs(self, point, outputs, id, info, source):
        print(
            f"FLEXIBLE: {len(self._flex_x)},{len(self._flex_y)},{len(self._flex_c)} | SIZE : {self._lf.pqueue.qsize()} <= {self.threshold}"
        )
        self._flex_x.append(point)
        self._flex_y.append(outputs["objective"])
        if self._lf.constraint is not None:
            self._flex_c.append([outputs[k] for k in self._lf.constraint])

        if self._lf.pqueue.qsize() <= self.threshold:
            return False  # Stop master
        else:
            return True  # Continue master


# Multi Synchronous -> Save score into groups, return groups to meta worker
class _MultiSynchronous_strat(_Parallel_strat):
    # send a point to loss master
    def _send_to_master(self, point):
        self.comm.send(dest=self.master_rank, tag=2, obj=point)

    # Executed by Experiment to compute X
    def __call__(self, X, stop_obj=None, **kwargs):
        # Early stopping
        ctn = True

        # score
        y = [None] * len(X)

        if self._lf.constraint is None:
            list_constraints = None
        else:
            list_constraints = np.ones((len(X), len(self._lf.constraint)), dtype=float)

        # send point, point ID and point info
        for i, p in enumerate(X):
            self._send_to_master([p, i, kwargs, self._lf.rank])  # send point

        nb_recv = 0
        while nb_recv < len(X) and ctn:
            # receive score from loss
            logger.debug(f"call() of rank :{self._lf.rank} receiveing message")
            msg = self.comm.recv(source=self.master_rank, status=self._lf.status)
            tag = self._lf.status.Get_tag()

            if tag == 9:
                logger.debug(f"call() of rank :{self._lf.rank} |!| STOPPING |!|")
                ctn = False
                X, y = None, None
            elif tag == 2:
                logger.debug(f"call() of rank :{self._lf.rank} received a score")
                # id / score
                y[msg[1]] = msg[0]
                if self._lf.constraints is not None:
                    list_constraints[msg[1]] = msg[2]  # type: ignore
                nb_recv += 1

        return X, y, list_constraints

    # Executed by master when it receives a score from a worker
    def _process_outputs(self, point, outputs, id, info, source):
        if self._lf.constraint is not None:
            constraints = [outputs[k] for k in self._lf.constraint]
        else:
            constraints = []
        self.comm.send(
            dest=source,
            tag=2,
            obj=(outputs["objective"], id, constraints),
        )
        return True


# Multi Asynchronous -> Return unique score vers worker meta
class _MultiAsynchronous_strat(_Parallel_strat):
    def __init__(self, loss, master_rank):
        super().__init__(loss, master_rank)
        self._current_points = {}

    # send a point to loss master
    def _send_to_master(self, point, infos):
        id = self._computed  # number of computed points used as id
        self._current_points[id] = point
        self.comm.send(dest=self.master_rank, tag=2, obj=(point, id, infos))
        self._computed += 1

    # Executed by Experiment to compute X
    def __call__(self, X, stop_obj=None, **kwargs):
        # send point, point ID and point info
        for p in X:
            self._send_to_master(p, kwargs)  # send point

        # receive score from loss
        logger.info(f"call() of rank :{self._lf.rank} receiveing message")
        msg = self.comm.recv(source=self.master_rank, status=self._lf.status)
        tag = self._lf.status.Get_tag()
        source = self._lf.status.Get_source()

        new_x, y = None, None

        if tag == 9:
            logger.info(f"call() of rank :{self._lf.rank} |!| STOPPING |!|")
            new_x, y = None, None
        elif tag == 2:
            logger.info(f"call() of rank :{self._lf.rank} received a score")
            # id / score
            new_x, y = self._current_points.pop(msg[1]), msg[0]  # type: ignore
        else:
            raise ValueError(
                "Unknown tag message in _MultiAsynchronous_strat, got tag={tag}"
            )

        logger.info("RETURN: ", [new_x], [y])

        if self._lf.constraint is None:
            return [new_x], [y], None
        else:
            return [new_x], [y], [msg[2]]

    # Executed by master when it receives a score from a worker
    def _process_outputs(self, point, outputs, id, info, source):
        if self._lf.constraint is not None:
            constraints = [outputs[k] for k in self._lf.constraint]
        else:
            constraints = []
        self.comm.send(
            dest=source,
            tag=2,
            obj=(outputs["objective"], id, constraints),
        )
        return True


# Wrap different loss functions
def Loss(
    model=None,
    objective=Minimizer,
    save: Optional[str] = None,
    record_time=False,
    MPI=False,
    only_score=False,
    kwargs_mode=False,
    workers=None,
    default=None,
    constraint=None,
    **kwargs,
):
    """Loss

    Wrap a function of type :math:`f(x)=y`. See :ref:`lf` for more info.

    Parameters
    ----------
    model : Callable
        Function of type :math:`f(x)=y`. With :math:`x` a solution, a set
        of hyperparameters for example.
        And :math:`y` can be a single value, a list, a tuple, or a dict,
        containing the loss value and other optionnal information.
        It can also be of mixed types, containing, strings, float, int...
    objective : Objective, default=Minimizer
        An :code:`Objective` object determines what the optimization problem is.
        If :code:`objective` is :code:`Maximizer`, then the first argument
        of the object, list, tuple or dict, returned by the :code:`__call__`
        function will be maximized.
    save : str, optionnal
        If a :code:`str` is given, then outputs will be saved in :code:`save`.
    record_time : boolean, default=False
            If True, :code:`start_time`, :code:`end_time`, :code:`start_date`, :code:`end_date` will be recorded
            and saved in the save file for each :code:`__call__`.
    only_score : bool, default=False
        If True, then only the score of evaluated solutions are saved.
        Otherwise, all infos returned by the :ref:`lf` and :ref:`meta` are
        saved.
    kwargs_mode : bool, default=False
        If True, then solutions are passed as kwargs to :ref:`lf`. Keys are
        the names of the :ref:`var` within the :ref:`sp`.
    MPI : {False, 'asynchronous', 'synchronous', 'flexible'}, optional
        Wrap the function with :code:`MPILoss` if True, with SerialLoss else.
        if :code:`strategy='synchronous`: then :code:`__call__` will return all results from all
        solutions passed, once all of them have been evaluated.
        if :code:`strategy='asynchronous`: then :code:`__call__` will return
        the result from an evaluation of a solution assoon as it receives a
        result from a worker. Other solutions, are still being evaluated in
        background.
        if :code:`strategy='flexible`: then :code:`__call__` will return
        all computed results, only if the number of remaining uncomputed solutions
        is below a certain threshold. Pass: :code:`threshold=int` kwarg, to :code:`Loss`
        or :code:`MPILoss`.
    workers : int, optionnal
        Number of workers among the total number of processes spawned by
        MPI. At least, one process is dedicated to the master.
    default : dict, optionnal
        Dictionnary of defaults arguments, kwargs, to pass to the loss function.
        They are not affected by any :ref:`metaheuristic` or other methods.
    constraint : list[str], default=None
            Constraints works when the model returns a dictionnary of values.
            Constraints values returned by the model must be booleans.
            If a list of strings is passed, constraints values will be passed to
            the :code:`forward` method of :ref:`meta`.

    Returns
    -------
    wrapper : LossFunc
        Wrapped original function

    Examples
    --------
    >>> import numpy as np
    >>> from zellij.core.loss_func import Loss
    >>> @Loss(save=False, verbose=True)
    ... def himmelblau(x):
    ...   x_ar = np.array(x)
    ...   return np.sum(x_ar**4 -16*x_ar**2 + 5*x_ar) * (1/len(x_ar))
    >>> print(f"Best solution found: f({himmelblau.best_point}) = {himmelblau.best_score}")
    Best solution found: f(None) = inf
    >>> print(f"Number of evaluations:{himmelblau.calls}")
    Number of evaluations:0
    """
    if model:
        return SerialLoss(model)
    else:

        def wrapper(model):
            if MPI:
                return MPILoss(
                    model,
                    objective,
                    save,
                    record_time,
                    only_score,
                    kwargs_mode,
                    workers=workers,
                    strategy=MPI,  # type: ignore
                    default=default,
                    constraint=constraint,
                    **kwargs,
                )
            else:
                return SerialLoss(
                    model,
                    objective,
                    save,
                    record_time,
                    only_score,
                    kwargs_mode,
                    default=default,
                    constraint=constraint,
                    **kwargs,
                )

        return wrapper


class MockModel(object):
    """MockModel

    This object allows to replace your real model with a costless object,
    by mimicking different available configurations in Zellij.
    ** Be carefull: This object does not replace any Loss wrapper**

    Parameters
    ----------
    outputs : dict, default={'o1',lambda *args, **kwargs: np.random.random()}
        Dictionnary containing outputs name (keys)
        and functions to execute to obtain outputs.
        Pass *args and **kwargs to these functions when calling this MockModel.

    verbose : bool
        If True logger.info information when saving and __call___.

    return_format : string
        Output format. It can be :code:`'dict'` > :code:`{'o1':value1,'o2':value2,...}`
        or :code:`list`>:code:`[value1,value2,...]`.

    return_model : boolean
        Return :code:`(outputs, MockModel)` if True. Else, :code:`outputs`.

    See Also
    --------
    Loss : Wrapper function
    MPILoss : Distributed version of LossFunc
    SerialLoss : Basic version of LossFunc

    Examples
    --------
    >>> from zellij.core.loss_func import MockModel, Loss
    >>> mock = MockModel()
    >>> logger.info(mock("test", 1, 2.0, param1="Mock", param2=True))
    I am Mock !
        ->*args: ('test', 1, 2.0)
        ->**kwargs: {'param1': 'Mock', 'param2': True}
    ({'o1': 0.3440051802032301},
    <zellij.core.loss_func.MockModel at 0x7f5c8027a100>)
    >>> loss = Loss(save=True, verbose=False)(mock)
    >>> logger.info(loss([["test", 1, 2.0, "Mock", True]], other_info="Hi !"))
    I am Mock !
        ->*args: (['test', 1, 2.0, 'Mock', True],)
        ->**kwargs: {}
    I am Mock !
        ->saving in MockModel_zlj_save/model/MockModel_best/i_am_mock.txt
    [0.7762604280531996]
    """

    def __init__(
        self,
        outputs={"o1": lambda *args, **kwargs: np.random.random()},
        return_format="dict",
        return_model=True,
        verbose=True,
    ):
        super().__init__()
        self.outputs = outputs
        self.return_format = return_format
        self.return_model = return_model
        self.verbose = verbose

    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        filename = os.path.join(filepath, "i_am_mock.txt")
        with open(filename, "wb") as f:
            if self.verbose:
                logger.info(f"\nI am Mock !\n\t->saving in {filename}")

    def __call__(self, *args, **kwargs):
        if self.verbose:
            logger.info(f"\nI am Mock !\n\t->*args: {args}\n\t->**kwargs: {kwargs}")

        if self.return_format == "dict":
            part_1 = {x: y(*args, **kwargs) for x, y in self.outputs.items()}
        elif self.return_format == "list":
            part_1 = [y(*args, **kwargs) for x, y in self.outputs.items()]
        else:
            raise NotImplementedError(
                f"return_format={self.return_format} is not implemented"
            )
        if self.return_model:
            return part_1, self
        else:
            return part_1
