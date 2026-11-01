import unittest
import numpy as np

from smt_optim.acquisition_functions import (
    mean_ivr,
    vec_mean_ivr,
)


class TestMeanIVR(unittest.TestCase):
    def test_mean_ivr(self):

        mu = 1.0
        ivr = 0.5
        np.testing.assert_array_equal(mean_ivr(mu, ivr), np.array([1.0, -0.5]))

    def test_vec_mean_ivr(self):

        mu = np.array([[1.0], [2.0]])
        ivr = np.array([[0.5], [0.0]])

        vec_vals = vec_mean_ivr(mu, ivr)

        # test output shape
        self.assertEqual(vec_vals.shape[0], 2)
        self.assertEqual(vec_vals.shape[1], 2)

        # test that vec_mean_ivr corresponds to mean_ivr
        np.testing.assert_array_equal(vec_vals[0], mean_ivr(mu[0, 0], ivr[0, 0]))


if __name__ == "__main__":
    unittest.main()
