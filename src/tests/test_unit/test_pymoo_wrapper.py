import unittest

import numpy as np

from smt_optim.benchmarks.base import PymooWrapper
from smt_optim.core import ConstraintConfig, ObjectiveConfig, Problem
from smt_optim.core.state import State
from smt_optim.surrogate_models.smt import SmtAutoModel
from smt_optim.utils.multi_obj import PymooStateWrapper

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV


class TestParetoFront(unittest.TestCase):
    def test_constrained_benchmark(self):
        from smt_optim.benchmarks.multiobj.zdt_mf import MF_DTLZ5

        ref_point = np.array([2, 2])
        hv = HV(ref_point=ref_point)

        problem = MF_DTLZ5()
        problem.set_dim(4)

        problem.num_cstr = 0
        problem.constraints = []

        pymoo_prob = PymooWrapper(problem)

        algorithm = NSGA2(pop_size=100, seed=0)
        res_uncstr = minimize(pymoo_prob, algorithm, ("n_gen", 100), seed=0)

        problem = MF_DTLZ5()
        problem.set_dim(4)

        pymoo_prob = PymooWrapper(problem)

        algorithm = NSGA2(pop_size=100, seed=0)
        res_cstr = minimize(pymoo_prob, algorithm, ("n_gen", 100), seed=0)

        self.assertLess(hv.do(res_cstr.F), hv.do(res_uncstr.F))


class LinearModel:
    def predict_values(self, x):
        return x[:, :1].copy()


class TestPymooStateWrapper(unittest.TestCase):
    def test_lower_and_upper_bounds(self):
        problem = Problem(
            obj_configs=[ObjectiveConfig([lambda x: 0.0], SmtAutoModel)],
            design_space=np.array([[0.0, 1.0], [0.0, 1.0]]),
            cstr_configs=[
                ConstraintConfig(
                    [lambda x: 0.0], lower=0.2, upper=0.6, surrogate=SmtAutoModel
                )
            ],
        )
        state = State(problem)
        state.obj_models = [LinearModel()]
        state.cstr_models = [LinearModel()]
        # scaled bounds, normally set when the dataset is scaled
        state.cstr_lower = np.array([0.2])
        state.cstr_upper = np.array([0.6])

        pymoo_prob = PymooStateWrapper(state, scaled=True, train=False)

        x = np.array([[0.0, 0.5], [0.4, 0.5], [0.9, 0.5]])
        out = dict()
        pymoo_prob._evaluate(x, out)

        # one column per bound, feasible when G <= 0
        np.testing.assert_allclose(out["G"][:, 0], 0.2 - x[:, 0])
        np.testing.assert_allclose(out["G"][:, 1], x[:, 0] - 0.6)

        # only the middle point satisfies both bounds
        np.testing.assert_array_equal(
            np.all(out["G"] <= 0, axis=1), [False, True, False]
        )


if __name__ == "__main__":
    unittest.main()
