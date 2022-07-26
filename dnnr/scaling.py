"""This module contains the input scaling."""

import abc
import dataclasses
import random as random_mod
import warnings
from typing import Any, Optional, Union

import numpy as np
import scipy.optimize
import scipy.spatial.distance

# from sklearn.metrics import mean_absolute_error, mean_squared_error
import sklearn.metrics as sk_metrics
import tqdm.auto as tqdm
from sklearn import model_selection

import dnnr
from dnnr import nn_index


class InputScaling(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns the scaling vector of the input.

        Args:
            X_train: The training data.
            y_train: The training targets.
            X_test: The test data.
            y_test: The test targets.

        Returns:
            The scaling vector.
        """

    @abc.abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the input.

        Args:
            X: The input.

        Returns:
            The transformed input.
        """


class Identity(InputScaling):
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return np.ones(X_train.shape[1])

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class _Optimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, gradients: list[np.ndarray]) -> None:
        """Updates the parameters.

        Args:
            gradients: The gradients of the parameters.
        """


@dataclasses.dataclass
class SGD(_Optimizer):
    """Stochastic gradient descent optimizer.

    Args:
        parameters: The parameters to optimize.
        lr: The learning rate.
    """

    parameters: list[np.ndarray]
    lr: float = 0.01

    def step(self, gradients: list[np.ndarray]) -> None:
        for param, grad in zip(self.parameters, gradients):
            param -= self.lr * grad


@dataclasses.dataclass
class RMSPROP:
    """The RMSPROP optimizer.

    Args:
        parameters: The parameters to optimize.
        lr: The learning rate.
        γ: The decay rate.
        eps: The epsilon to avoid division by zero.
    """

    parameters: list[np.ndarray]
    lr: float = 1e-4
    γ: float = 0.99
    eps: float = 1e-08

    def __post_init__(self):
        self.v = [np.zeros_like(param) for param in self.parameters]

    def step(self, gradients: list[np.ndarray]) -> None:
        for param, grad, v in zip(self.parameters, gradients, self.v):
            # inplace update
            v[:] = self.γ * v + (1 - self.γ) * grad**2
            update = self.lr * grad / (np.sqrt(v) + self.eps)
            param -= update


@dataclasses.dataclass
class NumpyInputScaling(InputScaling):
    """This class handles the scaling of the input.

    Args:
        n_epochs: The number of epochs to train the scaling.
        optimizer: The optimizer to use (either `SGD` or `RMSPROP`).
        optimizer_params: The parameters of the optimizer.
        epsilon: The epsilon for gradient computation.
        random: The `random.Random` instance for this class.
        n_trees: Number of trees in the Annoy index
        show_progress: Whether to show a progress bar.
        fail_on_nan: Whether to fail on NaN values.
    """

    n_epochs: int = 1
    optimizer: Union[str, type[_Optimizer]] = SGD
    optimizer_params: dict[str, Any] = dataclasses.field(default_factory=dict)
    shuffle: bool = True
    epsilon: float = 1e-6
    random: random_mod.Random = dataclasses.field(
        default_factory=lambda: random_mod.Random(
            random_mod.randint(0, 2**32 - 1)
        )
    )
    n_trees: int = 25  # Number of trees in the Annoy index
    show_progress: bool = False
    fail_on_nan: bool = False
    index: Union[str, type[nn_index.BaseIndex]] = 'annoy'
    index_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.scaling_: Optional[np.ndarray] = None
        self.scaling_history: list = []
        self.scores_history: list = []
        self.costs_history: list = []
        self.index_cls = nn_index.get_index_class(self.index)
        self._fitted: bool = False

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self.scaling_ is None:
            raise RuntimeError("Not fitted")
        return X * self.scaling_

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        val_size: Optional[int] = None,
    ) -> np.ndarray:
        """Fits the scaling vector.

        Args:
            X_train: The training data.
            y_train: The training targets.
            X_val: The validation data.
            y_val: The validation targets.
            val_size: The size of the validation set.

        If the validation set is not provided, the training set is split into
        a validation set using the `val_size` parameter.

        Returns:
            The scaling vector.
        """

        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be either given or not.")

        if X_val is None and y_val is None:
            split_size = (
                val_size if val_size is not None else int(0.1 * len(X_train))
            )
            if split_size < 10:
                raise ValueError("Split size too small")
            X_train, X_val, y_train, y_val = model_selection.train_test_split(
                X_train,
                y_train,
                test_size=split_size,
                random_state=self.random.randint(0, 2**32 - 1),
            )

        assert X_val is not None
        assert y_val is not None

        def score():
            assert X_val is not None
            n_approx = min(int(X_train.shape[0] / 2), X_train.shape[1] * 6)
            model = dnnr.DNNR(n_approx=n_approx)
            model.fit(scaling * X_train, y_train)
            return sk_metrics.r2_score(y_val, model.predict(scaling * X_val))

        def handle_possible_nans(grad: np.ndarray) -> bool:
            if not np.isfinite(grad).all():
                if self.fail_on_nan:
                    raise RuntimeError("Gradient contains NaN or Inf")

                warnings.warn(
                    "Found inf/nans in gradient. " "Scaling is returned now."
                )

                self.scaling_ = self.scaling_history[
                    np.argmax(self.scores_history)
                ]
                return True
            else:
                return False

        def get_optimizer() -> _Optimizer:
            if isinstance(self.optimizer, str):
                optimizer_cls = {
                    'sgd': SGD,
                    'rmsprop': RMSPROP,
                }[self.optimizer.lower()]
            else:
                optimizer_cls = self.optimizer

            kwargs = self.optimizer_params.copy()
            kwargs['parameters'] = scaling
            return optimizer_cls(**kwargs)

        if self._fitted:
            raise RuntimeError("Already fitted scaling vector")

        self._fitted = True

        n_features = X_train.shape[1]
        batch_size = 8 * n_features
        scaling = np.ones((1, n_features))

        optimizer = get_optimizer()

        self.scaling_history.append(scaling.copy())
        self.scores_history.append(score())
        for epoch in tqdm.trange(self.n_epochs, disable=not self.show_progress):
            index = self.index_cls.build(scaling * X_train, **self.index_kwargs)

            train_index = list(range(len(X_train)))
            if self.shuffle:
                self.random.shuffle(train_index)
            for idx in train_index:
                v = X_train[idx]
                y = y_train[idx]
                indices, _ = index.query_knn(v * scaling[0], batch_size)
                # skip `v` itself
                indices = indices[1:]
                nn_x = X_train[indices]
                nn_y = y_train[indices]

                cost, grad = self._get_gradient(scaling, nn_x, nn_y, v, y)

                if handle_possible_nans(grad):
                    self.scaling_ = scaling
                    return self.scaling_

                self.costs_history.append(cost)
                optimizer.step([grad])

            self.scaling_history.append(scaling.copy())
            self.scores_history.append(score())

        best_scaling = self.scaling_history[np.argmax(self.scores_history)]
        self.scaling_ = best_scaling
        return best_scaling

    def _get_gradient(
        self,
        scaling: np.ndarray,
        nn_x: np.ndarray,
        nn_y: np.ndarray,
        v: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the loss and the gradient.

        Args:
            scaling: The scaling vector.
            nn_x: The nearest neighbors of the current sample.
            nn_y: The targets of the nearest neighbors.
            v: The current sample.
            y: The target of the current sample.
        """
        q = nn_y - y
        delta = nn_x - v
        try:
            pinv = np.linalg.pinv(delta.T @ delta)
            nn_y_hat = pinv @ (delta.T @ q)
        except RuntimeError:
            raise RuntimeError(
                "Failed to compute psydo inverse!"
                f" The scaling vector was: {scaling}"
            )

        y_pred = y + delta @ nn_y_hat.T
        scaled_nn_x = nn_x * scaling
        scaled_v = v * scaling

        h_norm_in = scaled_nn_x - scaled_v
        h = np.clip(np.linalg.norm(h_norm_in, axis=1), self.epsilon, None)

        q = np.abs(nn_y - y_pred)

        vq = q - np.mean(q)
        vh = h - np.mean(h)

        cossim = self._cossim(vq, vh)
        cost = -cossim
        # Backward path

        dcossim = -np.ones(1)  # ensure to account for - cossim
        _, dvh = self._cossim_backward(dcossim, cossim, vq, vh)

        # Derive: vh = h - np.mean(h)
        # d vh_j / d h_i =  - 1 / len(h)  if i != j
        # d vh_j / d h_i =  1 - 1 / len(h) if i == j
        #  -> I - 1/len(h)
        len_h = np.prod(h.shape)
        dim = dvh.shape[0]
        mean_len_matrix = np.full(dim, dim, 1 / len_h)
        mean_jac = np.eye(dim) - mean_len_matrix
        # dh = (1. - 1 / mean_len) * dvh
        dh = mean_jac @ dvh

        dh_norm_in = self._l2_norm_backward(dh, h, h_norm_in)

        # Derive: h_norm_in = scaled_nn_x - scaled_v
        dscaled_nn_x = dh_norm_in
        dscaled_v = -dh_norm_in

        # Derive: scaled_nn_x = nn_x * fsv
        dfsv_nn_x = nn_x * dscaled_nn_x
        # Derive: scaled_v = v * fsv
        dfsv_v = v * dscaled_v

        # Accumulate gradients
        dfsv = dfsv_nn_x + dfsv_v
        return cost, dfsv.sum(axis=0)

    @staticmethod
    def _l2_norm_backward(
        grad: np.ndarray, l2_norm: np.ndarray, a: np.ndarray
    ) -> np.ndarray:
        """Backward pass for the l2 norm.

        Args:
            grad: The backpropaged gradient.
            l2_norm: The l2 norm of the input.
            a: The input to the l2 norm.
        """
        # From: https://en.wikipedia.org/wiki/Norm_(mathematics)
        # d(||a||_2) / da = a / ||a||_2
        da = a / l2_norm[:, np.newaxis]
        return da * grad[:, np.newaxis]

    @staticmethod
    def _cossim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Computes the cosine similarity between two vectors."""
        return 1 - scipy.spatial.distance.cosine(a, b)

    @staticmethod
    def _cossim_backward(
        grad: np.ndarray,
        cossim: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        eps: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Backward pass for the cosine similarity.

        Args:
            grad: The backpropaged gradient.
            cossim: The cosine similarity of the input.
            a: The first input to the cosine similarity.
            b: The second input to the cosine similarity.
            eps: The epsilon to avoid numerical issues.

        Returns:
            A tuple of the gradient of the first input and the gradient of the
            second input.
        """
        # From: https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosine-similarity  # noqa
        #
        # d/da_i cossim(a, b) = b_i / (|a| |b|) - cossim(a, b) * a_i / |a|^2
        # analogously for b
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        dcos_da = (b / (na * nb + eps)) - (cossim * a / (na**2))
        dcos_db = (a / (na * nb + eps)) - (cossim * b / (nb**2))
        return dcos_da * grad, dcos_db * grad