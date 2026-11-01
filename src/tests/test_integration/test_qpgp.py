import unittest

import numpy as np

from smt_optim.core import Problem
from smt_optim.core import ObjectiveConfig, DriverConfig, Driver

from smt_optim.surrogate_models.smt import SmtAutoModel

from smt_optim.acquisition_strategies import qPGP
from smt_optim.acquisition_strategies.qpgp import select_batch


def branin(x: np.ndarray) -> float:
    x = np.atleast_2d(x)
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    value = (
        a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - r) ** 2
        + s * (1 - t) * np.cos(x[:, 0])
        + s
    )
    return value.item()


class TestqPGP(unittest.TestCase):
    def setUp(self):
        self.bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])

    def run_driver(self, batch_size, max_iter):
        obj_config = ObjectiveConfig(objective=[branin], surrogate=SmtAutoModel)

        problem = Problem(
            obj_configs=[obj_config],
            design_space=self.bounds,
        )

        driver = Driver(
            problem=problem,
            config=DriverConfig(max_iter=max_iter, nt_init=6, seed=42),
            strategy=qPGP,
            strategy_kwargs=dict(
                batch_size=batch_size, pop_size=20, n_gen=10, n_integration=64
            ),
        )

        return driver.optimize()

    def test_qpgp_batch_size(self):
        batch_size = 3
        max_iter = 2

        state = self.run_driver(batch_size, max_iter)

        # each iteration adds batch_size samples
        self.assertEqual(len(state.dataset.samples), 6 + batch_size * max_iter)

    def test_qpgp_branin(self):
        state = self.run_driver(batch_size=3, max_iter=10)

        f_min = min(sample.obj[0] for sample in state.dataset.samples)

        # global minimum of Branin: 0.398
        self.assertLessEqual(f_min, 1.0)

    def test_select_batch(self):
        rng = np.random.default_rng(0)
        pareto_x = rng.random((30, 2))
        pareto_mu = rng.random(30)
        best = pareto_x[np.argmin(pareto_mu)]

        batch = select_batch(pareto_x, pareto_mu, 5, seed=0)

        # batch_size distinct points from the Pareto set, lowest mean point first
        self.assertEqual(batch.shape, (5, 2))
        self.assertEqual(np.unique(batch, axis=0).shape[0], 5)
        for x in batch:
            self.assertTrue(np.any(np.all(np.isclose(pareto_x, x), axis=1)))
        np.testing.assert_allclose(batch[0], best)

        # batch_size = 1 -> lowest mean point only
        batch = select_batch(pareto_x, pareto_mu, 1, seed=0)
        self.assertEqual(batch.shape, (1, 2))
        np.testing.assert_allclose(batch[0], best)

        # fewer Pareto points than batch_size -> return all of them, lowest mean first
        batch = select_batch(pareto_x[:3], pareto_mu[:3], 5)
        self.assertEqual(batch.shape, (3, 2))
        np.testing.assert_allclose(batch[0], pareto_x[np.argmin(pareto_mu[:3])])

        # duplicates of the lowest mean point are not selected twice
        dup_x = np.vstack([pareto_x, best, best])
        dup_mu = np.concatenate([pareto_mu, [1.0, 1.0]])
        batch = select_batch(dup_x, dup_mu, 5, seed=0)
        self.assertEqual(np.unique(batch, axis=0).shape[0], 5)


if __name__ == "__main__":
    unittest.main()
