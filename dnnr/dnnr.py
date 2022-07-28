from __future__ import annotations

import dataclasses
from typing import Any, Optional, Union

import numpy as np
import sklearn.base

from dnnr import nn_index
from dnnr import scaling as scaling_mod
from dnnr import solver as solver_mod
from dnnr.solver import create_solver


@dataclasses.dataclass
class DNNR(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """DNNR model class.

    metric: distance metric used in the nearest neighbor index

    # Order of the Approximation

    The order of approximation can be controlled with the `order` argument:

    - `1`: Uses first-order approximation (the gradient)
    - `2`: The first-order and full second-order matrix (gradient & Hessian)
    - `2diag`: First-order and diagonal of the second-order derivatives
    - `3diag`: First-order and diagonals of the second and third-orders


    Args:
        n_neighbors: number of nearest neighbors to use.
        n_approx: number of neighbors used in approximating the derivatives.
        order: Taylor approximation order, one of `1`, `2`, `2diag`, `3diag`.
        fit_intercept: if True, the intercept is estimated. Otherwise, the
            point's ground truth label is used.
        solver: name of the equation solver used to approximate the derivatives.
        index: name of the index to be used for nearest neighbor (`annoy` or
            `kd_tree`).
        index_kwargs: keyword arguments to be passed to the index constructor.
        scaling: name of the scaling method to be used.
        scaling_kwargs: keyword arguments to be passed to the scaling method.
        precompute_derivatives: if True, the gradient is computed for each
            training point during the `fit`. Otherwise, the gradient is computed
            during the prediction.
        clipping: whether to clip the predicted output to the maximum and
            minimum of the target values of the train set: `[y_min, y_max]`.

    """

    # TODO: allow any metric that scipy.spatial.distance.pdist likes
    # TODO: allow any solver that scipy.optimize.minimize likes
    # TODO: define an index interface

    n_neighbors: int = 3
    n_approx: int = 32
    order: str = "1"
    fit_intercept: bool = False
    solver: Union[str, solver_mod.Solver] = "linear_regression"
    index: Union[str, nn_index.BaseIndex] = "annoy"
    index_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    scaling: Union[None, str, scaling_mod.InputScaling] = "learned"
    scaling_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    precompute_derivatives: bool = False
    clipping: bool = False

    def __post_init__(self):
        self.nn_index: nn_index.BaseIndex

        if isinstance(self.index, str):
            index_cls = nn_index.get_index_class(self.index)
            self.nn_index = index_cls(**self.index_kwargs)
        else:
            self.nn_index = self.index

        self.derivatives_: Optional[list[np.ndarray]] = None

        self._check_valid_order(self.order)

        self.fitted_ = False

    def _precompute_derivatives(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        """Computes the gradient for the training points and their estimated
        label from the taylor expansion

        Args:
            X_train (np.ndarray) with shape (n_samples, n_features)
            y_train (np.ndarray) with shape (n_samples, 1)
        """
        self.derivatives_ = []

        for v in X_train:
            indices, _ = self.nn_index.query_knn(v, self.n_approx + 1)
            # ignore the first index as its the queried point itself
            indices = indices[1:]

            self.derivatives_.append(
                self._estimate_derivatives(X_train[indices], y_train[indices])
            )

    def _get_scaler(self) -> scaling_mod.InputScaling:
        """Returns the scaler object"""
        if self.scaling is None:
            return scaling_mod.NoScaling()
        elif isinstance(self.scaling, str):
            if self.scaling == "None":
                return scaling_mod.NoScaling()
            elif self.scaling == "learned":
                return scaling_mod.NumpyInputScaling(**self.scaling_kwargs)
            else:
                raise ValueError("Unknown scaling method")
        else:
            return self.scaling

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> DNNR:

        # save dataset shapes
        m, n = X_train.shape
        self.n = n
        self.m = m

        if isinstance(self.solver, str):
            self.solver_ = create_solver(self.solver)
        else:
            self.solver_ = self.solver
        # create and build the nearest neighbors index

        self.scaling_ = self.scaling
        # save a copy of the training data, should be only used
        # with precompute_derivatives=False
        self.max_y = np.max(y_train)
        self.min_y = np.min(y_train)

        self.scaler_ = self._get_scaler()
        # scale the training data
        self.X_train = self.scaler_.fit_transform(X_train, y_train)
        del X_train
        self.y_train = y_train

        self.nn_index.fit(self.X_train, **self.index_kwargs)

        if self.precompute_derivatives:
            self._precompute_derivatives(self.X_train, y_train)

        self.fitted_ = True
        return self

    def _check_valid_order(self, order: str) -> None:
        if order not in ["1", "2", "2diag", "3diag"]:
            raise ValueError(
                "Unknown order. Must be one of `1`, `2`, `2diag`, `3diag`"
            )

    def _estimate_derivatives(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_neighbors: Optional[int] = None,
        order: Optional[str] = None,
    ) -> np.ndarray:
        def _create_2der_mat(mat: np.ndarray) -> np.ndarray:
            """Creates 2-order matrix."""

            der_mat = np.zeros((mat.shape[0], mat.shape[1] ** 2))
            for i in range(mat.shape[0]):
                der_mat[0, :] = (
                    mat[i].reshape(-1, 1) @ mat[i].reshape(-1, 1).T
                ).reshape(-1)
            return der_mat

        nn_indices, _ = self.nn_index.query_knn(x, n_neighbors or self.n_approx)
        deltas_1st = self.X_train[nn_indices] - x
        ys = self.y_train[nn_indices] - y
        order = order or self.order
        self._check_valid_order(order)

        if self.fit_intercept:
            deltas_1st = np.concatenate(
                [deltas_1st, np.ones((deltas_1st.shape[0], 1))], axis=1
            )

        # take care of higher order terms
        if "1" == order:
            deltas = deltas_1st
        elif "2diag" == order:
            deltas_2nd = 0.5 * np.power(self.X_train[nn_indices] - x, 2)
            deltas = np.concatenate([deltas_1st, deltas_2nd], axis=1)
        elif "2" == order:
            deltas_2nd = _create_2der_mat(self.X_train[nn_indices] - x)
            deltas = np.concatenate([deltas_1st, deltas_2nd], axis=1)
        elif "3diag" == order:
            deltas_2nd = 0.5 * np.power(self.X_train[nn_indices] - x, 2)
            deltas_3rd = (1 / 6) * np.power(self.X_train[nn_indices] - x, 3)
            deltas = np.concatenate(
                [deltas_1st, deltas_2nd, deltas_3rd], axis=1
            )
        else:
            raise ValueError(f"Unknown order: {order}")

        w = np.ones(deltas.shape[0])
        # solve for the gradients nn_y_hat
        gamma = self.solver_.solve(deltas, ys, w)
        return gamma

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("DNNR is not fitted! Call `.fit()` first.")

        predictions = []
        for v in self.scaler_.transform(X_test):
            indices, _ = self.nn_index.query_knn(v, self.n_neighbors)
            predictions_of_neighbors = []
            for i in range(self.n_neighbors):
                # get the neighbor's neighbors' features and labels
                nn = self.X_train[indices[i]]
                nn_y = self.y_train[indices[i]]
                # gamma contains all estimated derivatives
                if self.derivatives_ is not None:
                    gamma = self.derivatives_[int(indices[i])]
                else:
                    gamma = self._estimate_derivatives(nn, nn_y)

                intercept = nn_y if "x" not in self.order else gamma[self.n]

                x_delta = v - nn

                # perform taylor approximation to predict the point's label
                prediction = intercept + gamma[: self.n].dot(x_delta)
                offset = 1 if "x" in self.order else 0
                # take care of higher order terms:
                if "2diag" in self.order:
                    nn_y_hat_2nd = gamma[self.n + offset :]
                    prediction += nn_y_hat_2nd.dot(0.5 * (np.power(x_delta, 2)))
                elif "2" in self.order:
                    nn_y_hat_2nd = gamma[self.n + offset :]
                    nn_y_hat_2nd = nn_y_hat_2nd.reshape(self.n, self.n)
                    prediction += 0.5 * (x_delta).T.dot(nn_y_hat_2nd).dot(
                        x_delta
                    )
                elif "3diag" in self.order:
                    nn_y_hat_2nd = gamma[self.n + offset : 2 * self.n + offset]
                    nn_y_hat_3rd = gamma[2 * self.n + offset :]
                    prediction = (
                        prediction
                        + nn_y_hat_2nd.dot(0.5 * (np.power(x_delta, 2)))
                        + nn_y_hat_3rd.dot((1 / 6) * (np.power(x_delta, 3)))
                    )
                predictions_of_neighbors.append(prediction)
            predictions.append(np.mean(predictions_of_neighbors))
        if self.clipping:
            return np.clip(predictions, a_min=self.min_y, a_max=self.max_y)
        return np.array(predictions)

    def point_analysis(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        gradient_scaling_vector: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, ...]:
        index = 0
        feature_importances = []
        distance_importances = []
        point_class = []
        r2s = []
        for v in X_test:
            indices, _ = self.nn_index.query_knn(v, self.n_neighbors)
            indices = indices[1:]
            for i in range(self.n_neighbors - 1):
                # get the neighbhor's features and label
                nn = self.X_train[indices[i]]
                nn_y = self.y_train[indices[i]]
                # get the neighbhors of this neighbhor
                nn_indices, _ = self.nn_index.query_knn(nn, self.n_approx)
                nn_indices = nn_indices[1:]  # drop the neighbor itself
                # Δx = X_{nn} - X_{i}
                x_deltas = self.X_train[nn_indices] - nn
                y_deltas = self.y_train[nn_indices] - nn_y
                w = np.ones(x_deltas.shape[0])
                # solve for the gradient estimate γ: γ @ Δx =  Δy
                gamma = self.solver_.solve(x_deltas, y_deltas, w)
                local_pred = nn_y + x_deltas @ gamma.T
                SS_res = np.mean(
                    np.power(local_pred - self.y_train[nn_indices], 2)
                )
                SS_tot = np.mean(
                    np.power(
                        self.y_train[nn_indices]
                        - np.mean(self.y_train[nn_indices]),
                        2,
                    )
                )
                R2 = 1 - (SS_res / SS_tot)
                r2s.append(R2)
                # perform taylor expansion to predict the point's label
                delta_pred = v - nn
                intercept = nn_y if "x" not in self.order else gamma[self.n]
                prediction = intercept + gamma[: self.n].dot(delta_pred)
                p_class = np.abs(prediction - y_test[index])
                feat_imp = np.multiply(gamma[: self.n], delta_pred)
                feature_importances.append(feat_imp)
                point_class.append(p_class)
                distance_importances.append(v - nn)
            index += 1
        return (
            np.array(feature_importances),
            np.array(point_class),
            np.array(distance_importances),
            np.array(r2s),
        )

    def __repr__(self) -> str:
        return "DNNR(n_neighbors={n_neighbors},n_approx={n_approx})".format(
            n_neighbors=self.n_neighbors, n_approx=self.n_approx
        )

    def __str__(self) -> str:
        return "instance of DNNR"
