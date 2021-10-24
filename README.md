# multilabel_knn
[![Build Status](https://travis-ci.org/skojaku/multilabel_knn.svg?branch=main)](https://travis-ci.org/skojaku/multilabel_knn)
[![Unit Test & Deploy](https://github.com/skojaku/multilabel_knn/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/multilabel_knn/actions/workflows/main.yml)


`multilabel_knn` is a lightweight toolbox for the multilabel classifications based on the k-nearest neighbor graphs.

###

The following algorithms are implemented:

- k-nearest neighbor classifier
- [multilabel k-nearest neighbor classifier](https://linkinghub.elsevier.com/retrieve/pii/S0031320307000027)
- [binomial multilabel k-nearest neighbor classifier](see here)
- [binomial multilabel graph neighbor classifer](see here)

## Requirements
- Python 3.7 or later

## Doc

https://multilabel_knn.readthedocs.io/en/latest/

## Install

```bash
pip install multilabel_knn
```

`multilabel_knn` uses [faiss library](https://github.com/facebookresearch/faiss), which has two versions, `faiss-cpu` and `faiss-gpu`.
As the name stands, `faiss-gpu` can leverage GPUs, thureby faster if you have GPUs. `multilabel_knn` uses `faiss-cpu` by default to avoid unnecessary GPU-related troubles.
But, if you have gpus compatible with the `faiss-gpu`, you can benefit the gpu accelarations by installing `faiss-gpu` by
you can still leverage the GPUs (which is recommended if you have) by installing

*with conda*:
```bash
conda install -c conda-forge faiss-gpu
```

or *with pip*:
```
pip install faiss-gpu
```

*Don't forget to pass `gpu_id` to the `init` argument to enable GPU*


## Maintenance

Code Linting:
```bash
conda install -y -c conda-forge pre-commit
pre-commit install
```

Docsctring: sphinx format

Test:
```bash
python -m unittest tests/simple_test.py
```
