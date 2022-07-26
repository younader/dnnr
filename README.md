# dnnr


[![Build Status](https://github.com/younader/dnnr/actions/workflows/dev.yml/badge.svg)](https://github.com/younader/dnnr/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/younader/dnnr/branch/main/graphs/badge.svg)](https://codecov.io/github/younader/dnnr)



Easy to use package of the DNNR regression


* Documentation: <https://younader.github.io/dnnr>
* GitHub: <https://github.com/younader/dnnr>
* PyPI: <https://pypi.org/project/dnnr/>
* Free software: MIT



## Setup


### Get it running


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
python -m dnnr nodes         # now you can list the available nodes
make test                                               # or run the tests
```
