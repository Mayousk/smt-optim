import numpy as np


def positive_part(x: float | np.ndarray) -> float | np.ndarray:
    """Return max(x, 0)."""
    return np.maximum(x, 0.0)


# Backward compatibility alias
PositivePart = positive_part


def init_bi_obj_composite_ei(state, kwargs):
    """
    Initialize the composite expected improvement acquisition function.
    """
    phi = kwargs["phi"]
    n_expectancy = kwargs.get("n_accuracy", 1000)
    seed = kwargs.get("seed", getattr(state, "seed", None))

    models = state.obj_models
    training_outputs = state.scaled_dataset.export_data([0, 1], 0)
    f_min = min(phi(y) for y in training_outputs)

    # Use deterministic RNG when seed is provided to ensure smooth acquisition surface
    rng = np.random.default_rng(seed) if seed is not None else None

    def composite_expected_improvement(
        mu: np.ndarray, s: np.ndarray, f_min: float, n_expectancy: int = n_expectancy
    ) -> float:
        """
        Expected Improvement composite acquisition function.

        Parameters
        ----------
        mu : np.ndarray
            Mean prediction of shape (2,).
        s : np.ndarray
            Standard deviation prediction of shape (2,).
        f_min : float
            Best minimum objective value in training data.
        n_expectancy : int
            Number of Monte-Carlo samples.

        Returns
        -------
        float
            Expected Improvement value.
        """
        if np.all(s <= 1e-16):
            return float(positive_part(f_min - phi(mu)))

        if rng is not None:
            sample_z = rng.standard_normal((n_expectancy, 2))
        else:
            sample_z = np.random.normal(size=(n_expectancy, 2))

        # Vectorized generation of samples: shape (n_expectancy, 2)
        samples = mu + s * sample_z
        phi_values = np.fromiter(
            (phi(pt) for pt in samples), dtype=float, count=n_expectancy
        )
        improvements = positive_part(f_min - phi_values)
        return float(np.mean(improvements))

    def composite_ei(x_pred: np.ndarray) -> float:
        x_pred = np.atleast_2d(x_pred)
        var0 = max(
            1e-16, float(np.asarray(models[0].predict_variances(x_pred)).ravel()[0])
        )
        var1 = max(
            1e-16, float(np.asarray(models[1].predict_variances(x_pred)).ravel()[0])
        )

        s = np.array([np.sqrt(var0), np.sqrt(var1)])
        y = np.array(
            [
                float(np.asarray(models[0].predict_values(x_pred)).ravel()[0]),
                float(np.asarray(models[1].predict_values(x_pred)).ravel()[0]),
            ]
        )
        return composite_expected_improvement(y, s, f_min)

    return composite_ei
