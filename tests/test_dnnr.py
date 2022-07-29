import random

import numpy as np

import dnnr


def test_model() -> None:
    np.random.seed(0)
    x = np.random.uniform(size=(500, 10))
    w = np.random.normal(size=(10, 1))
    y = x @ w
    y = y[:, 0]
    model = dnnr.DNNR(scaling=None)
    model.fit(x, y)
    diff = np.abs(model.predict(x) - y).mean()
    assert diff < 1


def test_kd_index() -> None:
    np.random.seed(0)
    x = np.random.uniform(size=(500, 10))
    w = np.random.normal(size=(10, 1))
    y = x @ w
    y = y[:, 0]
    model = dnnr.DNNR(index='kd_tree')
    model.fit(x, y)
    diff = np.abs(model.predict(x) - y).mean()
    assert diff < 1


def test_dnnr_scaling() -> None:
    np.random.seed(0)
    x = np.random.normal(size=(500, 10))
    w = 0.2 * np.random.normal(size=(10, 1)) + 2.0
    # makes the last 5 dimensions unimportant
    w[5:] = 0
    y = x @ w + (0.2 * x @ w) ** 2
    y = y[:, 0]
    model = dnnr.DNNR(
        scaling='learned',
        scaling_kwargs=dict(
            n_epochs=10,
            random=random.Random(1),
        ),
    )
    model.fit(x, y)
    diff = np.abs(model.predict(x) - y).mean()
    assert diff < 1

    assert isinstance(model.scaler_, dnnr.LearnedScaling)
    assert model.scaler_.scaling_ is not None
    assert model.scaler_._fitted

    # unimported dimension are scaled lower
    assert (model.scaler_.scaling_[0, 5:] < 0.5).all()
    # imported dimension are scaled higher
    assert (model.scaler_.scaling_[0, :5] > 0.5).all()


def test_dnnr_scaling_low_samples() -> None:
    np.random.seed(0)
    x = np.random.normal(size=(20, 10))
    w = 0.2 * np.random.normal(size=(10, 1)) + 2.0
    y = x @ w + (0.2 * x @ w) ** 2
    # makes the last 5 dimensions unimportant
    model = dnnr.DNNR(scaling='learned', scaling_kwargs=dict(n_epochs=10))
    model.fit(x, y)
    assert np.allclose(
        model.scaler_.scaling_, np.ones_like(model.scaler_.scaling_)
    )


def test_dnnr_higher_orders() -> None:
    np.random.seed(0)
    x = np.random.normal(size=(500, 10))
    w = 0.2 * np.random.normal(size=(10, 1)) + 2.0
    # makes the last 5 dimensions unimportant
    w[5:] = 0
    y = x @ w + (0.2 * x @ w) ** 2
    y = y[:, 0]

    for order in ['2diag', '3diag', '2']:
        model = dnnr.DNNR(
            order=order, n_neighbors=4, n_derivative_neighbors=50, clip=True
        )
        model.fit(x, y)
        diff = np.abs(model.predict(x[:20]) - y[:20]).mean()
        if order != "2":
            assert diff < 1


def test_dnnr_point_analysis() -> None:
    np.random.seed(0)
    x = np.random.normal(size=(500, 10))
    w = 0.2 * np.random.normal(size=(10, 1)) + 2.0
    # makes the last 5 dimensions unimportant
    w[5:] = 0
    y = x @ w + (0.2 * x @ w) ** 2
    y = y[:, 0]

    model = dnnr.DNNR(order='2diag')
    model.fit(x, y)
    results = model.point_analysis(x[:3])

    assert isinstance(results[0], dnnr.dnnr.DNNRPrediction)
