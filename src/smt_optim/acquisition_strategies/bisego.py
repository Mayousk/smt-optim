import numpy as np
from scipy import stats
from scipy.stats import norm

from smt.surrogate_models import KRG
from smt_optim.acquisition_functions import log_ei
from smt_optim.acquisition_functions.composite_expected_improvement import (
    init_bi_obj_composite_ei,
)
from smt_optim.acquisition_strategies import AcquisitionStrategy
from smt_optim.core.state import State
from smt_optim.subsolvers import multistart_minimize


def dominates(p, q):
    """Return True if point p strictly dominates point q, else False."""
    return (p[0] <= q[0] and p[1] < q[1]) or (p[0] < q[0] and p[1] <= q[1])


def pareto_front(d, y):
    """Given a DoE (d, y), return the list of indices of non-dominated points sorted by ascending f1."""
    t = len(d)
    front = []
    for i in range(t):
        if all(not dominates(q, y[i]) for q in y):
            front.append((y[i][0], i))
    front.sort()
    return [p[1] for p in front]


def positive_part(x):
    """Return max(x, 0)."""
    return max(x, 0.0)


def single_objective_normalized(y, r, s=None):
    """Return single-objective normalized formulation."""
    if s is None:
        s = [1.0] * len(y)
    return max((y[i] - r[i]) / s[i] for i in range(len(y)))


def single_objective_product(y, r):
    """Return single-objective product formulation."""
    return -float(np.prod([positive_part(r[i] - y[i]) ** 2 for i in range(len(y))]))


def norm_2(p):
    """Return 2-norm of point p."""
    return float(np.sqrt(sum(x**2 for x in p)))


def dist(p, q):
    """Return Euclidean distance between two points."""
    return norm_2([p[i] - q[i] for i in range(len(p))])


def dist_to_neighbors(p1, p2, p3, w):
    """Return sum of squared distances from p2 to neighbors p1 and p3, weighted by w."""
    return (dist(p1, p2) ** 2 + dist(p2, p3) ** 2) / (w + 1)


# Backward compatibility aliases
Dominates = dominates
ParetoFront = pareto_front
PositivePart = positive_part
SingleObjectiveNormalized = single_objective_normalized
SingleObjectiveProduct = single_objective_product
Norm = norm_2
Dist = dist
DistToNeighbors = dist_to_neighbors


class BiSEGO(AcquisitionStrategy):
    """
    Bi-objective Scalarized Efficient Global Optimization

    This acquisition strategy can perform bi-objective optimization on unconstrained optimization problems.

    Parameters
    ----------
    state : State
        Optimization state containing the current problem definition,
        surrogate models, and dataset.

    acq_func : acquisition function, optional
        Acquisition function used for the Pareto front initialization.
        Default is 'log_ei'.

    acq_func_bi : acquisition function, optional
        Acquisition function used for the composite case.
        Default is 'init_bi_obj_composite_ei'.

    acq_func_naive : acquisition function, optional
        Acquisition function used for the naive case.
        Default is 'log_ei'.

    n_multi_start : int, optional
        Number of starting points used for local acquisition optimization.
        Default is 10.

    n_accuracy : int, optional
        Number of Monte-Carlo calls for the composite acquisition function
        Default is 100

    sp_method : str, optional
        Local optimization method used for acquisition refinement.
        Default is "SLSQP".

    sp_tol : float, optional
        Tolerance for the acquisition function optimizer. Default is
        `sqrt(np.finfo(float).eps)`.

    so_formulation : str, optional
        Choice of single-objective formulation. "Product" or "Normalized". The normalized
        scalarisation is not compatible with the naive approach.
        Default is "Product".

    single_obj_max_calls : int, optional
        Maximum number of calls using the same single-objective formulation
        Default is 'self.n_multi_start'.

    init_calls : int, optional
        Number of calls for the initial optimization process of f1 and f2 independently.
        Default is 'self.n_multi_start'.

    naive : bool, optional
        Choice of the version of the algorithm. If False, the composite expected improvement function is used. If True, log_ei is used.
        Default is False.

    verbose : bool, optional
        If True, prints the different steps of the algorithm as it runs.
        Default is False

    """

    def __init__(self, state: State, **kwargs):
        super().__init__()

        self.acq_func1 = kwargs.get("acq_func", log_ei)
        self.acq_func2 = kwargs.get("acq_func", log_ei)
        self.acq_func_gen3 = kwargs.get("acq_func_bi", init_bi_obj_composite_ei)
        self.n_multi_start = kwargs.pop("n_multi_start", 10)
        self.n_accuracy = kwargs.pop("n_accuracy", 100)
        self.sp_method = kwargs.pop("sp_method", "Cobyla")
        self.sp_tol = kwargs.pop("sp_tol", np.sqrt(np.finfo(float).eps))
        self.so_formulation = kwargs.pop(
            "so_formulation", kwargs.pop("soformulation", "Product")
        )
        self.current_calls = 0
        self.current_subcalls = 0
        self.single_obj_max_calls = kwargs.pop(
            "single_obj_max_calls", self.n_multi_start
        )
        self.init_calls = kwargs.pop("init_calls", self.n_multi_start)
        self.acq_func_gen1 = lambda state, kwargs: (
            lambda x: float(
                np.asarray(
                    self.acq_func1(
                        state.obj_models[0].predict_values(x),
                        state.obj_models[0].predict_variances(x),
                        float(np.min(state.scaled_dataset.export_data([0], 0))),
                    )
                ).ravel()[0]
            )
        )
        self.acq_func_gen2 = lambda state, kwargs: (
            lambda x: float(
                np.asarray(
                    self.acq_func2(
                        state.obj_models[1].predict_values(x),
                        state.obj_models[1].predict_variances(x),
                        float(np.min(state.scaled_dataset.export_data([1], 0))),
                    )
                ).ravel()[0]
            )
        )
        self.naive = kwargs.pop("naive", False)
        self.verbose = kwargs.pop("verbose", False)

        self.r = None
        self.state = state
        self.X = None
        self.W = None

    def validate_config(self, state):
        pass

    def get_scaled_DoE(self):
        Y = self.state.scaled_dataset.export_as_dict()["obj"]
        D = self.state.scaled_dataset.export_as_dict()["x"]
        return (D, Y)

    def get_DoE(self):
        Y = self.state.dataset.export_as_dict()["obj"]
        D = self.state.dataset.export_as_dict()["x"]
        return (D, Y)

    def get_pareto_front(self):
        D, Y = self.get_scaled_DoE()
        self.X = pareto_front(D, Y)

    def select_reference_point(self):
        self.get_pareto_front()
        J = len(self.X)
        D, Y = self.get_scaled_DoE()
        X = self.X
        W = self.W
        if J > 2:
            # Select a point of the Pareto front relatively far from its neighbors
            j = max(
                (dist_to_neighbors(Y[X[k - 1]], Y[X[k]], Y[X[k + 1]], W[X[k]]), k)
                for k in range(1, J - 1)
            )[1]
            r = (Y[X[j + 1]][0], Y[X[j - 1]][1])
        elif J == 2:
            j = 1
            r = (Y[X[1]][0], Y[X[0]][1])
        elif J == 1:
            return None
        else:
            raise ValueError("The Pareto Front is empty")
        self.W[X[j]] += 1
        return r

    def get_infill(self, state):
        old_pareto_front = self.X
        self.get_pareto_front()

        if self.current_calls == 0:
            if self.current_calls < self.init_calls and self.verbose:
                print("Min(f1) phase")
            self.W = [0 for _ in range(len(self.state.dataset.export_as_dict()["x"]))]

        # Pre-treatment phase : find min(f1) and min(f2)
        if self.current_calls < self.init_calls:
            self.current_calls += 1
            self.W.append(0)
            return self.get_infill_custom(state, self.acq_func_gen1)
        elif self.current_calls < 2 * self.init_calls:
            if self.current_calls == self.init_calls and self.verbose:
                print("Min(f2) phase")
            self.current_calls += 1
            self.W.append(0)
            return self.get_infill_custom(state, self.acq_func_gen2)

        # Main loop
        else:
            if (
                self.current_calls == 2 * self.init_calls
                or self.current_subcalls == 0
                or self.current_subcalls == self.single_obj_max_calls
                or old_pareto_front != self.X
            ):
                if self.verbose:
                    print("The Pareto front is of length", len(self.X))
                self.current_subcalls = 0
                r = self.select_reference_point()
                if (
                    r is None
                ):  # edge case where there is only one point in the Pareto Front
                    self.current_subcalls += 1
                    self.current_calls += 1
                    self.W.append(0)
                    if self.current_calls % 2:
                        return self.get_infill_custom(state, self.acq_func_gen1)
                    else:
                        return self.get_infill_custom(state, self.acq_func_gen2)
                if self.verbose:
                    print("Bi-objective phase with r =", r)
                if self.so_formulation == "Normalized":
                    self.phi = lambda y: single_objective_normalized(y, r)
                elif self.so_formulation == "Product":
                    self.phi = lambda y: single_objective_product(y, r)
                else:
                    raise ValueError(
                        f"Unknown single-objective formulation: {self.so_formulation}"
                    )
            self.current_subcalls += 1
            self.current_calls += 1
            self.W.append(0)

            if not self.naive:
                return self.get_infill_custom(
                    state, self.acq_func_gen3, phi=self.phi, n_accuracy=self.n_accuracy
                )
            else:
                return self.get_infill_naive(state, self.phi)

    def get_infill_custom(self, state, acq_func_gen, **kwargs):
        # Gets the infill point using the custom acquisition function provided
        self.seed = state.iter
        kwargs["seed"] = self.seed

        sampler = stats.qmc.LatinHypercube(d=state.problem.num_dim, rng=state.iter)
        multi_x0 = sampler.random(self.n_multi_start)

        ac_func = acq_func_gen(state, kwargs)

        def sp_wrapper(x):
            x = x.reshape(1, -1)
            return -ac_func(x)

        res = multistart_minimize(
            sp_wrapper,
            bounds=np.array([[0, 1]] * state.problem.num_dim),
            constraints=[],
            n_start=self.n_multi_start,
            multi_x0=multi_x0,
            seed=self.seed,
            tol=self.sp_tol,
            method=self.sp_method,
        )

        next_x = res.x
        infill = [next_x.reshape(1, -1)]

        return infill

    def get_infill_naive(self, state, phi):
        D, Y = self.get_scaled_DoE()
        Phi_values = np.array([phi(y) for y in Y])

        # 1. Train model on DoE
        sm = KRG(print_global=False)
        sm.set_training_values(np.atleast_2d(D), Phi_values.reshape(-1, 1))
        sm.train()

        # Find the current minimum value in the scalarized history
        current_best = float(np.min(Phi_values))

        # Define the negative log Expected Improvement to minimize
        def log_ei_fn(x):
            x_reshaped = np.atleast_2d(x)
            mu = sm.predict_values(x_reshaped)[0, 0]
            sigma2 = sm.predict_variances(x_reshaped)[0, 0]
            sigma = np.sqrt(max(sigma2, 1e-16))
            improvement = current_best - mu
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            return np.log(max(ei, 1e-16))

        # 2. Find maximum of log_EI acquisition function
        self.seed = state.iter

        sampler = stats.qmc.LatinHypercube(d=state.problem.num_dim, rng=state.iter)
        multi_x0 = sampler.random(self.n_multi_start)

        def sp_wrapper(x):
            x = x.reshape(1, -1)
            return -log_ei_fn(x)

        res = multistart_minimize(
            sp_wrapper,
            bounds=np.array([[0, 1]] * state.problem.num_dim),
            constraints=[],
            n_start=self.n_multi_start,
            multi_x0=multi_x0,
            seed=self.seed,
            tol=self.sp_tol,
            method=self.sp_method,
        )

        next_x = res.x
        infill = [next_x.reshape(1, -1)]

        return infill
