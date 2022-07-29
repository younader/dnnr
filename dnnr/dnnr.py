from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional, Union

import numpy as np
import sklearn.base

from dnnr import nn_index
from dnnr import scaling as scaling_mod
from dnnr import solver as solver_mod
from dnnr.solver import create_solver


@dataclasses.dataclass
class NeighborPrediction:
    neighbor_x: np.ndarray
    neighbor_y: np.ndarray
    neighbors_xs: np.ndarray
    neighbors_ys: np.ndarray
    query: np.ndarray  # point to predict
    local_prediction: np.ndarray  # local prediction
    derivative: np.ndarray  # derivative used to predict the point
    prediction_fn: Callable[[np.ndarray], np.ndarray]
    intercept: Optional[np.ndarray] = None


@dataclasses.dataclass
class DNNRPrediction:
    query: np.ndarray
    y_pred: np.ndarray
    neighbor_predictions: list[NeighborPrediction]
    y_true: Optional[np.ndarray] = None


@dataclasses.dataclass
class DNNR(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """DNNR model class.

    metric: distance metric used in the nearest neighbor index

    # Order of the Approximation

    The order of approximation can be controlled with the `order` argument:

        - `1`: Uses first-order approximation (the gradient)
        - `2diag`: First-order and diagonal of the second-order derivatives
        - `2`: The first-order and full second-order matrix (gradient & Hessian)
        - `3diag`: First-order and diagonals of the second and third-orders

    The recommendation is to use `1` which is the most efficient one, using
    `2diag` can sometimes improve the performance. `2` and `3diag` are more
    expensive and usually also do not deliver a better performance.

    Args:
        n_neighbors: number of nearest neighbors to use. The default value of
            `3` is usually a good choice.
        n_derivative_neighbors: number of neighbors used in approximating the
            derivatives. As a default value, we choose `3 * dim` where `dim` is
            the dimension of the input data. This is usually a good heuristic,
            but we would recommend to use a hyperparameter search to find the
            best value for it.
        order: Taylor approximation order, one of `1`, `2`, `2diag`, `3diag`.
            The preferable option here is `1` and sometimes `2diag` can deliver
            small improvements. `2` and `3diag` are implemented but usually do
            not yield significant improvements.
        fit_intercept: if True, the intercept is estimated. Otherwise, the
            point's ground truth label is used.
        solver: name of the equation solver used to approximate the derivatives.
            As default `linear_regression` is used. Other options are
            `scipy_lsqr`, `numpy`, `ridge` and `lasso`. Also accepts any class
            inheriting from `dnnr.solver.Solver`.
        index: name of the index to be used for nearest neighbor (`annoy` or
            `kd_tree`). Also accepts any subclass of `dnnr.nn_index.BaseIndex`.
        index_kwargs: keyword arguments passed to the index constructor.
        scaling: name of the scaling method to be used. If it is `None` or
            `no_scaling`, the data is not scaled. If it is `learned`, the
            scaling is learned using the cosine similarity objective.
        scaling_kwargs: keyword arguments to be passed to the scaling method.
        precompute_derivatives: if True, the gradient is computed for each
            training point during the `fit`. Otherwise, the gradient is computed
            during the prediction.
        clip: whether to clip the predicted output to the maximum and
            minimum of the target values of the train set: `[y_min, y_max]`.
    """

    n_neighbors: int = 3
    n_derivative_neighbors: int = -1
    order: str = "1"
    fit_intercept: bool = False
    solver: Union[str, solver_mod.Solver] = "linear_regression"
    index: Union[str, nn_index.BaseIndex] = "annoy"
    index_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    scaling: Union[None, str, scaling_mod.InputScaling] = "learned"
    scaling_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    precompute_derivatives: bool = False
    clip: bool = False

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
            indices, _ = self.nn_index.query_knn(
                v, self.n_derivative_neighbors + 1
            )
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
            if self.scaling in ["None", 'no_scaling']:
                return scaling_mod.NoScaling()
            elif self.scaling == "learned":
                return scaling_mod.LearnedScaling(**self.scaling_kwargs)
            else:
                raise ValueError("Unknown scaling method")
        else:
            return self.scaling

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> DNNR:

        # save dataset shapes
        m, n = X_train.shape
        self.n = n
        self.m = m

        if self.n_derivative_neighbors == -1:
            self.n_derivative_neighbors = 3 * self.n

        if isinstance(self.solver, str):
            self.solver_ = create_solver(self.solver)
        else:
            self.solver_ = self.solver
        # create and build the nearest neighbors index

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

    def _compute_deltas(
        self, query: np.ndarray, xs: np.ndarray, order: str
    ) -> np.ndarray:
        self._check_valid_order(order)

        def _create_2der_mat(mat: np.ndarray) -> np.ndarray:
            """Creates 2-order matrix."""

            der_mat = np.zeros((mat.shape[0], mat.shape[1] ** 2))
            for i in range(mat.shape[0]):
                der_mat[0, :] = (
                    mat[i].reshape(-1, 1) @ mat[i].reshape(-1, 1).T
                ).reshape(-1)
            return der_mat

        deltas_1st = xs - query

        if self.fit_intercept:
            deltas_1st = np.concatenate(
                [deltas_1st, np.ones((deltas_1st.shape[0], 1))], axis=1
            )

        if "1" == order:
            deltas = deltas_1st
        # take care of higher order terms
        elif "2diag" == order:
            deltas_2nd = 0.5 * np.power(xs - query, 2)
            deltas = np.concatenate([deltas_1st, deltas_2nd], axis=1)
        elif "2" == order:
            deltas_2nd = _create_2der_mat(xs - query)
            deltas = np.concatenate([deltas_1st, deltas_2nd], axis=1)
        elif "3diag" == order:
            deltas_2nd = 0.5 * np.power(xs - query, 2)
            deltas_3rd = (1 / 6) * np.power(xs - query, 3)
            deltas = np.concatenate(
                [deltas_1st, deltas_2nd, deltas_3rd], axis=1
            )
        else:
            raise ValueError(f"Unknown order: {order}")
        return deltas

    def _estimate_derivatives(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_neighbors: Optional[int] = None,
        order: Optional[str] = None,
    ) -> np.ndarray:

        nn_indices, _ = self.nn_index.query_knn(
            x, n_neighbors or self.n_derivative_neighbors
        )
        ys = self.y_train[nn_indices] - y
        order = order or self.order

        deltas = self._compute_deltas(x, self.X_train[nn_indices], order)
        w = np.ones(deltas.shape[0])
        # solve for the gradients nn_y_hat
        gamma = self.solver_.solve(deltas, ys, w)
        return gamma

    def _compute_local_prediction(
        self,
        query: np.ndarray,
        neighbor: np.ndarray,
        derivatives: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        intercept = derivatives[self.n] if self.fit_intercept else y
        x_delta = query - neighbor

        # perform taylor approximation to predict the point's label
        prediction = intercept + derivatives[: self.n].dot(x_delta)
        offset = 1 if self.fit_intercept else 0
        # take care of higher order terms:
        if "2diag" in self.order:
            nn_y_hat_2nd = derivatives[self.n + offset :]
            prediction += nn_y_hat_2nd.dot(0.5 * (np.power(x_delta, 2)))
        elif "2" in self.order:
            nn_y_hat_2nd = derivatives[self.n + offset :]
            nn_y_hat_2nd = nn_y_hat_2nd.reshape(self.n, self.n)
            prediction += 0.5 * (x_delta).T.dot(nn_y_hat_2nd).dot(x_delta)
        elif "3diag" in self.order:
            nn_y_hat_2nd = derivatives[self.n + offset : 2 * self.n + offset]
            nn_y_hat_3rd = derivatives[2 * self.n + offset :]
            prediction = (
                prediction
                + nn_y_hat_2nd.dot(0.5 * (np.power(x_delta, 2)))
                + nn_y_hat_3rd.dot((1 / 6) * (np.power(x_delta, 3)))
            )
        return prediction

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

                prediction = self._compute_local_prediction(v, nn, gamma, nn_y)
                predictions_of_neighbors.append(prediction)
            predictions.append(np.mean(predictions_of_neighbors))
        if self.clip:
            return np.clip(predictions, a_min=self.min_y, a_max=self.max_y)
        return np.array(predictions)

    def point_analysis(
        self,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
    ) -> list[DNNRPrediction]:
        index = 0
        predictions = []
        for v in X_test:
            neighbors = []
            indices, _ = self.nn_index.query_knn(v, self.n_neighbors + 1)

            # if point is in the training set, we skip it
            if np.allclose(v, self.X_train[indices[0]]):
                indices = indices[1:]
            else:
                indices = indices[:-1]

            for i in range(self.n_neighbors - 1):
                # get the neighbhor's features and label
                nn = self.X_train[indices[i]]
                nn_y = self.y_train[indices[i]]
                # get the neighbhors of this neighbhor
                nn_indices, _ = self.nn_index.query_knn(
                    nn, self.n_derivative_neighbors
                )
                nn_indices = nn_indices[1:]  # drop the neighbor itself
                # Δx = X_{nn} - X_{i}
                gamma = self._estimate_derivatives(nn, nn_y)

                local_pred = self._compute_local_prediction(v, nn, gamma, nn_y)
                neighbor_pred = NeighborPrediction(
                    neighbor_x=nn,
                    neighbor_y=nn_y,
                    neighbors_xs=self.X_train[nn_indices],
                    neighbors_ys=self.y_train[nn_indices],
                    query=v,
                    local_prediction=local_pred,
                    derivative=gamma,
                    intercept=nn_y if self.fit_intercept else gamma[self.n],
                    prediction_fn=lambda query: self._compute_local_prediction(
                        query, self.X_train[nn_indices], gamma, nn_y
                    ),
                )
                neighbors.append(neighbor_pred)
            predictions.append(
                DNNRPrediction(
                    query=v,
                    y_pred=np.mean([n.local_prediction for n in neighbors]),
                    y_true=y_test[index] if y_test is not None else None,
                    neighbor_predictions=neighbors,
                )
            )
            index += 1
        return predictions
