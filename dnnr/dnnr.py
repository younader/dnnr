from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from dnnr.nn_index import create_index
from dnnr.solver import create_solver


class DNNR(BaseEstimator, RegressorMixin):
    """DNNR model class.

    Args:
        n_neighbors: number of nearest neighbors to use.
        n_approx: number of neighbors used in approximating the gradient
        mode: Taylor approximation mode, one of `1`, `2`, `2diag`, `2diag`, `3`.
        metric: distance metric used in the nearest neighbor index
        index: name of the index to be used for nearest neighbor (`annoy` or
            `kd_tree`).
        solver: name of the equation solver used in gradient computation.
    """

    # TODO: allow any metric that scipy.spatial.distance.pdist likes
    # TODO: allow any solver that scipy.optimize.minimize likes
    # TODO: define an index interface

    def __init__(
        self,
        n_neighbors: int = 3,
        n_approx: int = 32,
        mode: str = "1",
        metric: str = "euclidean",
        index: str = "annoy",
        solver: str = "lr",
        scaling: str = "None",
        precompute: bool = False,
        weighted: bool = False,
        clipping: bool = False,
    ) -> None:

        self.n_neighbors = n_neighbors
        self.n_approx = n_approx
        self.metric = metric
        self.index_name = index
        self.solver_name = solver
        self.precompute_gradients = precompute
        self.weighted = weighted
        self.gradients: list[np.ndarray] = []
        self.scaling = scaling
        self.mode = mode
        self.clipping = clipping

    def _precompute_gradients(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        """Computes the gradient for the training points and their estimated
        label from the taylor expansion

        Args:
            X_train (np.ndarray) with shape (n_samples, n_features)
            y_train (np.ndarray) with shape (n_samples, 1)
        """
        gradients = []
        estimated_labels = []
        for v in X_train:
            indices, _ = self.index.query_knn(v, self.n_approx)
            # ignore the first index as its the point itself
            indices = indices[1:]
            neighs = X_train[indices] - v
            m = LinearRegression(fit_intercept=False).fit(
                np.concatenate(
                    [
                        neighs,
                        np.array(
                            [
                                [1],
                            ]
                            * (self.n_approx - 1)
                        ),
                    ],
                    axis=1,
                ),
                y_train[indices],
            )
            nn_y_hat = m.coef_[: self.vector_length]
            gradients.append(nn_y_hat)
            estimated_labels.append(m.coef_[self.vector_length])
        self.gradients = gradients
        self.estimated_labels = estimated_labels

    def _get_gradients(self) -> list[np.ndarray]:
        """Returns the stored gradients for the training datapoints"""
        if self.gradients:
            return self.gradients
        else:
            raise ValueError("Gradients not yet computed.")

    def _get_estimated_labels(self) -> list[np.ndarray]:
        """Returns the stored estimated labels for the training datapoints"""
        if self.estimated_labels:
            return self.estimated_labels
        else:
            raise ValueError("Estimated labels not yet computed.")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> DNNR:
        # save dataset shapes
        m, n = X_train.shape
        self.n = n
        self.m = m
        # create solver object
        self.solver = create_solver(self.solver_name)
        # create and build the nearest neighbhors index
        self.index = create_index(self.index_name, self.metric, n)
        fsv = np.ones(n)
        if self.precompute_gradients:
            self._precompute_gradients(X_train, y_train)
        # save a copy of the training data, should be only used
        # with precompute_gradients=False
        self.X_train = X_train * fsv
        self.y_train = y_train
        self.max_y = np.max(y_train)
        self.min_y = np.min(y_train)
        self.index.build(self.X_train)
        self.fsv = fsv
        if self.weighted:
            raise NotImplementedError("Method not yet implemented")
        return self

    def _estimate_gradient(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_neighbors: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        def _create_2der_mat(mat: np.ndarray) -> np.ndarray:
            """Creates 2-order matrix."""

            der_mat = np.zeros((mat.shape[0], mat.shape[1] ** 2))
            for i in range(mat.shape[0]):
                der_mat[0, :] = (
                    mat[i].reshape(-1, 1) @ mat[i].reshape(-1, 1).T
                ).reshape(-1)
            return der_mat

        nn_indices, _ = self.index.query_knn(x, n_neighbors or self.n_approx)
        deltas = self.X_train[nn_indices] - x
        ys = self.y_train[nn_indices] - y
        mode = mode or self.mode

        if "diag" in mode:
            neighs_2nd = 0.5 * np.power(self.X_train[nn_indices] - x, 2)
            deltas = np.concatenate([deltas, neighs_2nd], axis=1)
        elif "2" in mode:
            neighs_2nd = _create_2der_mat(self.X_train[nn_indices] - x)
            deltas = np.concatenate([deltas, neighs_2nd], axis=1)
        elif "3" in mode:
            neighs_2nd = 0.5 * np.power(self.X_train[nn_indices] - x, 2)
            neighs_3rd = (1 / 6) * np.power(self.X_train[nn_indices] - x, 3)
            deltas = np.concatenate([deltas, neighs_2nd, neighs_3rd], axis=1)

        w = np.ones(deltas.shape[0])
        # solve for the gradients nn_y_hat
        nn_y_hat = self.solver.solve(deltas, ys, w)
        # return gradient
        return nn_y_hat[: self.n]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = []
        for v in X_test * self.fsv:
            indices, _ = self.index.query_knn(v, self.n_neighbors)
            neigh_preds = []
            for i in range(self.n_neighbors):
                # get the neighbhor's features and label
                nn = self.X_train[indices[i]]
                nn_y = self.y_train[indices[i]]
                nn_y_hat = self._estimate_gradient(nn, nn_y)

                # perform taylor expansion to predict the point's label
                delta_pred = v - nn
                intercept = nn_y if "x" not in self.mode else nn_y_hat[self.n]

                prediction = intercept + nn_y_hat[: self.n].dot(delta_pred)
                offset = 1 if "x" in self.mode else 0
                if "diag" in self.mode:
                    nn_y_hat_2nd = nn_y_hat[self.n + offset :]
                    prediction += nn_y_hat_2nd.dot(
                        0.5 * (np.power(delta_pred, 2))
                    )
                elif "2" in self.mode:
                    nn_y_hat_2nd = nn_y_hat[self.n + offset :]
                    nn_y_hat_2nd = nn_y_hat_2nd.reshape(self.n, self.n)
                    prediction += 0.5 * (delta_pred).T.dot(nn_y_hat_2nd).dot(
                        delta_pred
                    )
                elif "3" in self.mode:
                    nn_y_hat_2nd = nn_y_hat[
                        self.n + offset : 2 * self.n + offset
                    ]
                    nn_y_hat_3rd = nn_y_hat[2 * self.n + offset :]
                    prediction = (
                        prediction
                        + nn_y_hat_2nd.dot(0.5 * (np.power(delta_pred, 2)))
                        + nn_y_hat_3rd.dot((1 / 6) * (np.power(delta_pred, 3)))
                    )
                neigh_preds.append(prediction)
            predictions.append(np.mean(neigh_preds))
        if self.clipping:
            predictions = np.clip(
                predictions, a_min=self.min_y, a_max=self.max_y
            )
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
            indices, _ = self.index.query_knn(v, self.n_neighbors)
            indices = indices[1:]
            for i in range(self.n_neighbors - 1):
                # get the neighbhor's features and label
                nn = self.X_train[indices[i]]
                nn_y = self.y_train[indices[i]]
                # get the neighbhors of this neighbhor
                nn_indices, _ = self.index.query_knn(nn, self.n_approx)
                nn_indices = nn_indices[1:]
                deltas = self.X_train[nn_indices] - nn

                ys = self.y_train[nn_indices] - nn_y
                w = np.ones(deltas.shape[0])
                if self.weighted:
                    pass
                # solve for the gradients nn_y_hat
                nn_y_hat = self.solver.solve(deltas, ys, w)
                local_ly = nn_y + deltas @ nn_y_hat.T
                SS_res = np.mean(
                    np.power(local_ly - self.y_train[nn_indices], 2)
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
                intercept = nn_y if "x" not in self.mode else nn_y_hat[self.n]
                prediction = intercept + nn_y_hat[: self.n].dot(delta_pred)
                p_class = np.abs(prediction - y_test[index])
                feat_imp = np.multiply(nn_y_hat[: self.n], delta_pred)
                feature_importances.append(feat_imp)
                point_class.append(p_class)
                distance_importances.append(v - nn)
                # grads.append(np.linalg.norm(nn_y_hat-opt_grad))
                # friedman_gradients.append(opt_grad)
                # opt_grad_pred=intercept+np.array(opt_grad).dot(delta_pred)
                # opt_grad_errors.append(np.abs(opt_grad_pred-y_test[index]))
            index += 1
        return (
            np.array(feature_importances),
            np.array(point_class),
            np.array(distance_importances),
            np.array(r2s),
        )
