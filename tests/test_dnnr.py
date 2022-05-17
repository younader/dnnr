import numpy as np

from dnnr import dnnr


def test_model() -> None:
    x = np.random.uniform(size=(100, 10))
    w = np.random.normal(size=(10, 1))
    y = x @ w
    model = dnnr.DNNR()
    model.fit(x, y)
    diff = np.abs(model.predict(x) - y).mean()
    assert diff < 1
