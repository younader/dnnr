from dataclasses import dataclass

import annoy
import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from dnnr.dnnr import DNNR

try:
    import torch  # noqa

    pytorch_missing = False
except ImportError:
    pytorch_missing = True


def get_torch_grad_scaler_cls():
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
            n_approx = min(int(X_train.shape[0] / 2), X_train.shape[1] * 6)
            model = DNNR(n_approx=n_approx)
            model.fit(X_train, y_train)
            return mean_absolute_error(y_test, model.predict(X_test))

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

            vx = q - torch.mean(q)
            vy = h - torch.mean(h)
            loss = torch.nn.CosineSimilarity(dim=0, eps=1e-4)
            cost = loss(vx, vy) * -1

            cost.backward()

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
