import abc
from typing import Optional

import numpy as np
import scipy.optimize
import scipy.spatial.distance


class InputScaling(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_scaling(
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


class NumpyInputScaling(InputScaling):
    def get_scaling(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        scipy.optimize.minimize
        return np.std(X_train, axis=0)

    def get_gradient(
        self,
        fsv: np.ndarray,
        nn_x: np.ndarray,
        nn_y: np.ndarray,
        v: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = nn_y - y
        delta = nn_x - v
        try:
            pinv = np.linalg.pinv(delta.T @ delta)
            nn_y_hat = pinv @ (delta.T @ q)
        except RuntimeError:
            raise RuntimeError(f"fsv vector is {fsv}")

        y_pred = y + delta @ nn_y_hat.T
        scaled_nn_x = nn_x * fsv
        scaled_v = v * fsv

        h_norm_in = scaled_nn_x - scaled_v
        h = np.linalg.norm(h_norm_in, axis=1)

        q = np.abs(nn_y - y_pred)

        vq = q - np.mean(q)
        vh = h - np.mean(h)

        cossim = self.cossim(vq, vh)
        cost = -cossim
        # Backward path

        dcossim = -np.ones(1)  # ensure to account for - cossim
        _, dvh = self.cossim_backward(dcossim, cossim, vq, vh)

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

        dh_norm_in = self.l2_norm_backward(dh, h, h_norm_in)

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
    def l2_norm_backward(
        grad: np.ndarray, l2_norm: np.ndarray, a: np.ndarray
    ) -> np.ndarray:
        # From: https://en.wikipedia.org/wiki/Norm_(mathematics)
        # d(||a||_2) / da = a / ||a||_2
        da = a / l2_norm[:, np.newaxis]
        return da * grad[:, np.newaxis]

    @staticmethod
    def cossim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return 1 - scipy.spatial.distance.cosine(a, b)

    @staticmethod
    def cossim_backward(
        grad: np.ndarray,
        cossim: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        eps: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray]:
        # From: https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosine-similarity  # noqa
        #
        # d/da_i cossim(a, b) = b_i / (|a| |b|) - cossim(a, b) * a_i / |a|^2
        # analogously for b
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        dcos_da = (b / (na * nb + eps)) - (cossim * a / (na**2))
        dcos_db = (a / (na * nb + eps)) - (cossim * b / (nb**2))
        return dcos_da * grad, dcos_db * grad

    #
    # Adapted from PyTorch:
    #
    # See: https://github.com/pytorch/pytorch/blob/e0295f55b574ab4883e01d228c0cc4fdd3f8b111/torch/csrc/autograd/FunctionsManual.cpp#L1377  # noqa: E501
    #
    # Tensor pinv_backward(
    #   const Tensor& grad,
    #   const Tensor& pinvA,
    #   const Tensor& A
    # ) {
    #   at::NoTF32Guard disable_tf32;
    #   auto m = A.size(-2);
    #   auto n = A.size(-1);
    #   auto pinvAh = pinvA.mH();
    #   auto gradh = grad.mH();
    #   // optimization to produce matrices of the smallest dimension
    #   if (m <= n) {
    #     auto K = gradh.matmul(pinvA);
    #     auto KpinvAh = K.matmul(pinvAh);
    #     return - (pinvA.matmul(K)).mH()
    #            + KpinvAh - (A.matmul(pinvA)).matmul(KpinvAh)
    #            + (pinvAh.matmul(pinvA)).matmul(gradh - K.matmul(A));
    #   }
    #   else {
    #     auto K = pinvA.matmul(gradh);
    #     auto pinvAhK = pinvAh.matmul(K);
    #     return - (K.matmul(pinvA)).mH()
    #            + (gradh - A.matmul(K)).matmul(pinvA).matmul(pinvAh)
    #            + pinvAhK - pinvAhK.matmul(pinvA).matmul(A);
    #   }
    @staticmethod
    def pinv_backward(
        grad: np.ndarray, pinvA: np.ndarray, A: np.ndarray
    ) -> np.ndarray:
        m = A.shape[0]
        n = A.shape[1]
        pinvAh = pinvA.T
        gradh = grad.T
        # optimization to produce matrices of the smallest dimension
        if m <= n:
            K = gradh @ pinvA
            KpinvAh = K @ pinvAh
            return (
                -(pinvA @ K).T
                + KpinvAh
                - (A @ pinvA) @ KpinvAh
                + (pinvAh @ pinvA) @ (gradh - K @ A)
            )
        else:
            K = pinvA @ gradh
            pinvAhK = pinvAh @ K
            return (
                -(K @ pinvA).T
                + (gradh - A @ K).T @ pinvA @ pinvAh
                + pinvAhK
                - pinvAhK @ pinvA @ A
            )
