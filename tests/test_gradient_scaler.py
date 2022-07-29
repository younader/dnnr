import random
from dataclasses import dataclass
from typing import Tuple

import annoy
import numpy as np
import pytest
import sklearn.datasets
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_model_selection
import sklearn.preprocessing as sk_preprocessing

import dnnr
from dnnr import scaling
from dnnr.dnnr import DNNR

try:
    import torch  # noqa

    pytorch_missing = False
except ImportError:
    pytorch_missing = True


def get_torch_grad_scaler_cls():
    # This pytorch class is imported from the paper repo
    # and is used to check the gradient computed with numpy.
    @dataclass
    class GradientScaler:
        """
        Gradient Based optimization for the scaling vector with the objective of
        correlating the distance between datapoints to the distance between
        labels
        """

        lr: float = 1e-4
        epochs: int = 1

        def score(self, X_train, y_train, X_test, y_test):
            n_derivative_neighbors = min(
                int(X_train.shape[0] / 2), X_train.shape[1] * 6
            )
            model = DNNR(n_derivative_neighbors=n_derivative_neighbors)
            model.fit(X_train, y_train)
            return sk_metrics.ean_absolute_error(y_test, model.predict(X_test))

        def neigh_scale(self, X_train, y_train, X_test=None, y_test=None):
            fsv = torch.full(
                X_train.shape[1:],
                1.0,
                requires_grad=True,
            )
            torch.autograd.set_detect_anomaly(False)
            batch_size = 8 * X_train.shape[1]
            fsv_history = []
            score_history = []
            lr = self.lr
            optimizer = torch.optim.RMSprop([fsv], lr=lr, centered=True)
            vector_length = X_train.shape[1]
            Ntrees = 25
            for epoch in range(self.epochs):
                fsv_copy = fsv.clone().data.cpu().numpy()
                fsv_history.append(fsv_copy)
                score_history.append(
                    self.score(
                        X_train * fsv_copy, y_train, X_test * fsv_copy, y_test
                    )
                )
                t1 = annoy.AnnoyIndex(vector_length, metric="euclidean")
                for i, v in zip(range(len(X_train)), X_train * fsv_copy):
                    t1.add_item(i, v)
                t1.build(Ntrees)
                index = 0
                optimizer.zero_grad()
                for v in X_train:
                    indices = t1.get_nns_by_vector(v * fsv_copy, batch_size)
                    indices = indices[1:]
                    nn_x = X_train[indices]
                    nn_y = torch.Tensor(y_train[indices])
                    y = torch.Tensor([y_train[index]])
                    q = nn_y - y
                    v = torch.Tensor(v)
                    nn_x = torch.Tensor(nn_x)
                    delta = nn_x - v
                    try:
                        nn_y_hat = torch.linalg.pinv(delta.T @ delta) @ (
                            delta.T @ q
                        )
                    except RuntimeError:
                        print(index)
                        raise RuntimeError(f"fsv vector is {fsv}")
                    y_pred = torch.Tensor(y + delta @ nn_y_hat.T)

                    nn_x = nn_x.multiply(fsv)
                    v = v.multiply(fsv)
                    h = torch.linalg.norm(nn_x - v, dim=1)

                    q = torch.abs(nn_y - y_pred)

                    vx = q - torch.mean(q)
                    vy = h - torch.mean(h)
                    loss = torch.nn.CosineSimilarity(dim=0, eps=1e-4)
                    cost = loss(vx, vy) * -1

                    cost.backward()

                    if torch.any(torch.isnan(fsv.grad)):
                        print(epoch, index)
                        return fsv.clone().data.cpu().numpy()
                    optimizer.step()
                    optimizer.zero_grad()

                    index += 1
            self.history = fsv_history
            self.scores = score_history
            self.fsv = fsv_history[np.argmin(score_history)]
            return fsv_history[np.argmin(score_history)]

        def transform(self, X):
            if len(self.fsv.shape) > 1:
                return X @ self.fsv
            else:
                return X * self.fsv

        def train_step(
            self,
            fsv: torch.Tensor,
            nn_x: torch.Tensor,
            nn_y: torch.Tensor,
            v: torch.Tensor,
            y: torch.Tensor,
        ) -> None:
            q = nn_y - y
            delta = nn_x - v
            try:
                nn_y_hat = torch.linalg.pinv(delta.T @ delta) @ (delta.T @ q)
            except RuntimeError:
                raise RuntimeError(f"fsv vector is {fsv}")
            y_pred = torch.Tensor(y + delta @ nn_y_hat.T)

            nn_x = nn_x.multiply(fsv)
            v = v.multiply(fsv)
            h = torch.linalg.norm(nn_x - v, dim=1)

            q = torch.abs(nn_y - y_pred)

            vq = q - torch.mean(q)
            vh = h - torch.mean(h)
            loss = torch.nn.CosineSimilarity(dim=0, eps=1e-4)
            cost = loss(vq, vh) * -1

            cost.backward()
            return cost

    return GradientScaler


@pytest.mark.skipif(pytorch_missing, reason="no pytorch found")
def test_pytorch_grad_scaler():
    scaler_cls = get_torch_grad_scaler_cls()
    scaler = scaler_cls()

    channels = 10
    neighbors = 100
    y = torch.randn(1)
    v = torch.randn(1, channels)
    nn_x = torch.randn(neighbors, channels)
    nn_y = torch.randn(neighbors)
    fsv = torch.ones(1, channels, requires_grad=True)
    scaler.train_step(fsv, nn_x, nn_y, v, y)

    assert fsv.grad is not None


@pytest.mark.skipif(pytorch_missing, reason="no pytorch found")
def test_numpy_grad_scaler():
    scaler_cls = get_torch_grad_scaler_cls()
    scaler = scaler_cls()

    torch.manual_seed(10)
    channels = 3
    neighbors = 6
    y = torch.randn(1)
    v = torch.randn(1, channels)
    nn_x = torch.randn(neighbors, channels)
    nn_y = torch.randn(neighbors)
    fsv = torch.ones(1, channels, requires_grad=True)
    cost = scaler.train_step(fsv, nn_x, nn_y, v, y)

    np_scaler = scaling.LearnedScaling()
    np_cost, fsv_grad = np_scaler._get_gradient(
        fsv.detach().numpy(), nn_x.numpy(), nn_y.numpy(), v.numpy(), y.numpy()
    )

    assert np.allclose(cost.detach().numpy(), np_cost)

    print("Torch:")
    print(fsv.grad.tolist())
    print("Numpy:")
    print(fsv_grad.tolist())
    assert np.allclose(fsv.grad.numpy(), fsv_grad, atol=1e-5)


@pytest.mark.skipif(pytorch_missing, reason="no pytorch found")
def test_cossim_backward():
    torch.manual_seed(0)
    a = torch.randn(4, requires_grad=True)
    b = torch.randn(4, requires_grad=True)
    cossim = torch.cosine_similarity(a, b, dim=0)
    cossim.backward()

    a_np = a.detach().numpy()
    b_np = b.detach().numpy()
    cossim_np = scaling.LearnedScaling._cossim(a_np, b_np)
    a_grad, b_grad = scaling.LearnedScaling._cossim_backward(
        np.ones(1),
        cossim_np,
        a_np,
        b_np,
    )
    print('-' * 80)
    print('numpy grad: ')
    print(a_grad)
    print('-' * 80)
    print('torch grad: ')
    print(a.grad)

    assert np.allclose(a_grad, a.grad.numpy(), atol=1e-5)

    print('-' * 80)
    print('numpy grad: ')
    print(b_grad.tolist())
    print('-' * 80)
    print('torch grad: ')
    print(b.grad.tolist())
    assert np.allclose(b_grad, b.grad.numpy(), atol=1e-5)


def test_scaling_on_california():
    cali = sklearn.datasets.fetch_california_housing()
    target = cali.target
    std_scaler = sk_preprocessing.StandardScaler()

    data = std_scaler.fit_transform(cali.data)

    # speedup the test, only run on a subset of the data
    np.random.seed(0)
    subset = np.random.choice(data.shape[0], 5000, replace=False)
    data = data[subset]
    target = target[subset]

    X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(
        data, target, test_size=0.15, random_state=0
    )

    idenity_scaler = scaling.NoScaling()
    idenity_scaler.fit(X_train, y_train)
    vanilla_model = dnnr.DNNR(scaling=idenity_scaler)
    assert (idenity_scaler.transform(X_train) == X_train).all()
    vanilla_model.fit(idenity_scaler.transform(X_train), y_train)
    vanilla_r2 = vanilla_model.score(X_test, y_test)

    optimizers = [
        ('sgd', dict(lr=0.01)),
        ('rmsprop', dict(lr=1e-3)),
    ]
    for opt, opt_kwargs in optimizers:
        scaler = scaling.LearnedScaling(
            n_epochs=2,
            fail_on_nan=True,
            show_progress=True,
            random=random.Random(0),
            optimizer=opt,
            optimizer_params=opt_kwargs,
        )

        scaled_model = dnnr.DNNR(scaling=scaler)
        scaled_model.fit(X_train, y_train)
        scaled_r2 = scaled_model.score(X_test, y_test)

        print('-' * 80)
        print(opt, 'Scaling data')
        print(scaler.scaling_history)
        print('-' * 80)
        print(opt, 'Scores data')
        print(scaler.scores_history)
        print('-' * 80)
        print(opt, 'scaling', scaler.scaling_)
        print('-' * 80)
        print('Unscaled R2:', vanilla_r2)
        print(opt, 'Scaled R2:', scaled_r2)
        assert scaled_r2 > vanilla_r2


def test_scaling_nans():
    def get_nans(
        # self: scaling.LearnedScaling,
        fsv: np.ndarray,
        nn_x: np.ndarray,
        nn_y: np.ndarray,
        v: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return np.nan * fsv.sum(), np.nan * fsv

    scaler_should_fail = scaling.LearnedScaling(
        n_epochs=1,
        fail_on_nan=True,
        show_progress=True,
        random=random.Random(0),
        optimizer='sgd',
    )

    scaler_should_fail._get_gradient = get_nans  # type: ignore

    np.random.seed(0)
    data = np.random.normal(size=(200, 3))
    target = np.random.normal(size=(200,))

    with pytest.raises(RuntimeError):
        scaler_should_fail.fit(data, target)

    scaler_should_warn = scaling.LearnedScaling(
        n_epochs=1,
        fail_on_nan=False,
        show_progress=True,
        random=random.Random(0),
        optimizer='sgd',
    )
    scaler_should_warn._get_gradient = get_nans  # type: ignore

    with pytest.warns():
        scaler_should_warn.fit(data, target)
