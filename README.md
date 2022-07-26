# DNNR: Differential Nearest Neighbors Regression

[![Build Status](https://github.com/younader/dnnr/actions/workflows/dev.yml/badge.svg)](https://github.com/younader/dnnr/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/younader/dnnr/branch/main/graphs/badge.svg)](https://codecov.io/github/younader/dnnr)

Implementation of ["DNNR: Differential Nearest Neighbors Regression"](https://proceedings.mlr.press/v162/nader22a.html).

Whereas KNN regression only uses the averaged value, DNNR also uses the gradient or even higher-order derivatives.

# 🚀 Quickstart


To install this project, run:

```bash
mkdir dnnr                   # this directory will hold the code, data and venv
cd dnnr
python3 -m venv venv                                    # create and load the virtual environment
source venv/bin/activate
# create the data folder
mkdir data
git clone https://github.com/younader/dnnr.git
cd dnnr
pip install -U pip wheel poetry
poetry install

make test                                               # or run the tests
```

# 🎉 Example





# 📊 Hyperparameters



* Scaling
* `n_neighbors`
* ...

# 📄 Citation



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
