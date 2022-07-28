import numpy as np

from dnnr import solver as solver_mod


def test_solvers():
    solvers = [
        "linear_regression",
        "scipy_lsqr",
        "numpy",
        "ridge",
        "lasso",
    ]
    X = np.random.normal(size=(100, 10))
    w = np.random.normal(size=(10))
    y = X @ w

    for solver_name in solvers:
        solver = solver_mod.create_solver(solver_name)
        w_solved = solver.solve(X, y, w=np.ones_like(y))
        assert w_solved.shape == w.shape
        if solver_name not in ["lasso", "ridge"]:
            assert np.allclose(w_solved, w, atol=0.01)
