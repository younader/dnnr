# DNNR: Differential Nearest Neighbors Regression

[![Build Status](https://github.com/younader/dnnr/actions/workflows/dev.yml/badge.svg)](https://github.com/younader/dnnr/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/younader/dnnr/branch/main/graphs/badge.svg)](https://codecov.io/github/younader/dnnr)

Implementation of ["DNNR: Differential Nearest Neighbors Regression"](https://proceedings.mlr.press/v162/nader22a.html).

Whereas KNN regression only uses the averaged value, DNNR also uses the gradient or even higher-order derivatives:

![KNN and DNNR Overview Image](knn_dnnr_overview.png)

# 🚀 Quickstart


To install this project, run:

```bash
pip install dnnr
```



# 🎉 Example

```python
from dnnr import DNNR

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

model = DNNR(n_neighbors=1, n_approx=3)
model.fit(x,y)
model.predict([[1.5]])
```

# 📊 Hyperparameters



* `n_neighbors` : number of nearest neighbors to use.
* `n_approx` : number of neighbors used in approximating the gradient.
* `mode` : Taylor approximation order.
* `metric` : distance metric used in the nearest neighbor index.
* `index` : name of the index to be used for nearest neighbor.
* `solver` : name of the equation solver used in gradient computation.
* `scaling` : whether to use DNNR scaling.

#  🛠 Development Installation

```bash
mkdir dnnr             # this directory will hold the code, data and venv
cd dnnr
python3 -m venv venv     # create and load the virtual environment
source venv/bin/activate
# create the data folder
mkdir data
git clone https://github.com/younader/dnnr.git
cd dnnr
pip install -U pip wheel poetry
poetry install

make test                          # or run the tests
```

# 📄 Citation

If you use this library for a scientific publication, please use the following BibTex entry to cite our work:

```bibtex
@InProceedings{pmlr-v162-nader22a,
  title = 	 {{DNNR}: Differential Nearest Neighbors Regression},
  author =       {Nader, Youssef and Sixt, Leon and Landgraf, Tim},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {16296--16317},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/nader22a/nader22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/nader22a.html},
}
```
