# multilabel_knn
[![Unit Test & Deploy](https://github.com/skojaku/multilabel_knn/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/multilabel_knn/actions/workflows/main.yml)


`multilabel_knn` is a lightweight toolbox for the multilabel classifications based on the k-nearest neighbor algorithms [[Doc](https://multilabel_knn.readthedocs.io/en/latest/)].

The following algorithms are implemented:

- k-nearest neighbor classifier
- [multilabel k-nearest neighbor classifier](https://linkinghub.elsevier.com/retrieve/pii/S0031320307000027) (recommended for a small number of labels)
- Binomial multilabel k-nearest neighbor classifier (recommended for large dataset with many labels)
- Binomial multilabel graph neighbor classifer

## Usage

**k-nearest neighbor algorithm (Predict a single label per sample)**

```python
import multilabel_knn as mlk
model = mlk.kNN(k=10, metric = "cosine") #k: number of neighbors, metric: distance metric {"euclidean", "cosine"}
model.fit(X, Y) # X :2d feature vectors. Y: label matrix, where Y[i,k] = 1 if i has label k.
Y_pred = model.predict(X_test) # Y_pred[i,k] = 1 is i is predicted to have label k.
```

**mutilabel kNN (Can predict multiple labels per sample) [1]**

```python
import multilabel_knn as mlk
model = mlk.multilabel_kNN(k=10, metric = "cosine")
model.fit(X, Y)
Y_pred = model.predict(X_test) 
```

**Binomial mutilabel kNN (Can predict multiple labels per sample)**

```python
import multilabel_knn as mlk
model = mlk.binomial_multilabel_kNN(k=10, metric = "cosine")
model.fit(X, Y) 
Y_pred = model.predict(X_test) 
```
*Binomial multilabel kNN is a mobidifed version of multilabel kNN. It can perform well for data with a large number of samples and labels. 
See the docstring for details.*

**Binomial mutilabel graph (Take a graph as input. Can predict multiple labels per node)**

```python
import multilabel_knn as mlk
model = mlk.binomial_multilabel_graph()
model.fit(A, Y) # A is the adjacency matrix of the graph for training. A[i,j] =1 if node i has a link to node j. 
Y_pred = model.predict(B) # B is the adjacency matrix of the biparite network, where B[i,j] =1 if node i has a link to node j in the training graph.
```

## Evaluation metrics

`multilabel_knn` has several evaluation metrics for multilabel classifications:  

```python
from multilabel_knn import evaluations

# Y: label matrix. Y[i,k]=1 if i has label k
# Y_pred: predicted label. Y_pred[i,k] if i is predicted to have label k  
evaluations.micro_f1score(Y, Y_pred) # micro f1

evaluations.macro_f1score(Y, Y_pred) # macro f1

evaluations.micro_hamming_loss(Y, Y_pred) # micro hamming loss

evaluations.macro_hamming_loss(Y, Y_pred) # macro hamming loss

# Y_score: probability or likelihood that i has label k
evaluations.average_precision(Y, Y_score) # average precision

evaluations.auc_roc(Y, Y_score) # roc-auc
````


## Install

Requirements: Python 3.7 or later


```bash
pip install multilabel_knn
```

`multilabel_knn` uses [faiss library](https://github.com/facebookresearch/faiss), which has two versions, `faiss-cpu` and `faiss-gpu`.
As the name stands, `faiss-gpu` can leverage GPUs, thureby faster if you have GPUs. `multilabel_knn` uses `faiss-cpu` by default to avoid unnecessary GPU-related troubles.
But, if you have gpus compatible with the `faiss-gpu`, you can benefit the gpu accelarations by installing `faiss-gpu` by

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

## Reference
[1] Zhang, Min-Ling, and Zhi-Hua Zhou. 2007. “ML-KNN: A Lazy Learning Approach to Multi-Label Learning.” Pattern Recognition 40 (7): 2038–48.
