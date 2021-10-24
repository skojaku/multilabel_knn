"""Multilabel k-NN for binomial features.

This is variant of the Multilabel k-NN for binomial feature.
Instead of predicting the labels from the k-nearest neighbors, this
classifiers predicts from the neighbors of a graph.
"""
import numpy as np
from scipy import sparse
from numba import njit
from scipy import stats
from .binom_multilabel_knn import binom_multilabel_kNN, _count_neighbors


class binom_multilabel_graph(binom_multilabel_kNN):
    def __init__(self, alpha=1, beta=1, prior="sample"):
        """
        :params n_neighbors: number of neighbors
        :params alpha: hyperparameter for the prior
        :params beta: hyperparameter for the prior
        :param k: [description], defaults to 5
        """
        self.prior_type = prior
        self.alpha = alpha
        self.beta = beta

    # fit knn model
    def fit(self, A, Y):
        """Model fitting

        :param A: adjacency matrix
        :type A: scipy.sparse.csr_matrix
        :param Y: label matrix
        :type Y: scipy.sparse.csr_matrix
        :return: self
        :rtype: object
        """

        A, Y = self._homogenize(A, Y)

        self.Y = Y

        self.n_indexed_samples, self.n_labels = Y.shape[0], Y.shape[1]

        #
        # Calculate the prior probabilities
        #
        if self.prior_type == "sample":
            self.priors = (self.alpha + np.array(Y.sum(axis=0)).reshape(-1)) / (
                self.alpha + self.beta + self.n_indexed_samples
            )
        elif self.prior_type == "uniform":
            self.priors = np.ones(self.n_labels) * 0.5

        #
        # Calculate the posterior probabilities
        #
        # make knn graph
        self.p1, self.p0 = self.estimate_binomial_params(A, Y)
        return self

    def predict(self, B):
        """predict labels for new samples

        :param B: adjacency matrix for a bipartite network. B[i,j] =1 if new node i has a link to the training node  if new node i has a link to the training node j.
        :type B: scipy.sparse.csr_matrix
        :return: predicted labels Y. Y[i,k] = 1 if the new node i is labeled with k.
        :rtype: scipy.sparse.csr_matrix
        """

        B = self._homogenize(B)

        C = B @ self.Y
        samples, labels, count = sparse.find(C)
        count = count.astype(int)

        n_neighbors = np.array(B.sum(axis=1)).reshape(-1)

        # Calculate the posterior probabilities
        safelog = lambda x: np.log(np.maximum(x, 1e-12))

        log_likelihood_1 = safelog(self.priors[labels])
        log_likelihood_1 += stats.binom.logpmf(
            count, n_neighbors[samples], self.p1[labels]
        )
        log_likelihood_0 = safelog(1 - self.priors[labels])
        log_likelihood_0 += stats.binom.logpmf(
            count, n_neighbors[samples], self.p0[labels]
        )

        pred_positive = log_likelihood_1 > log_likelihood_0

        Ypred = sparse.csr_matrix(
            (pred_positive, (samples, labels)), shape=(B.shape[0], self.n_labels)
        )

        return Ypred

    def estimate_binomial_params(self, A, Y):
        """
        Calculate the conditional probability p for the binomial distribution.

        params: A: knn graph
        params: Y: label matrix
        return: p1, p0: conditional probability for the binomial distribution
        """
        n_nodes, n_labels = Y.shape[0], Y.shape[1]

        n_neighbors = np.array(A.sum(axis=1)).reshape(-1)

        Y.sort_indices()
        C1, Call = _count_neighbors(
            A.indptr, A.indices, Y.indptr, Y.indices, n_nodes, n_labels
        )
        B1 = np.array((sparse.diags(n_neighbors) @ Y).sum(axis=0)).reshape(-1)
        Ball = np.sum(n_neighbors)

        p1 = (self.alpha + C1) / (self.alpha + self.beta + B1)
        p0 = (self.alpha + Call - C1) / (self.alpha + self.beta + Ball - B1)
        return p1, p0

    def _homogenize(self, A, Y=None):
        """homogeinize the input data

        :param A: adjacency matrix
        :type A: scipy.sparse.csr_matrix or numpy.ndarray
        :param Y: label matrix, defaults to None
        :type Y: scipy.sparse.csr_matrix, optional
        :raises ValueError: if A is neither scipy sparse matrix nor numpy array
        :raises ValueError: if Y is neither scipy sparse matrix nor numpy array
        :return: A, Y
        :rtype: scipy.sparse.csr_matrix, scipy.sparse.csr_matrix
        """
        if sparse.issparse(A):
            if not sparse.isspmatrix_csr(A):
                A = sparse.csr_matrix(A)
        elif isinstance(A, np.ndarray):
            A = sparse.csr_matrix(A)
        else:
            raise ValueError("A must be a scipy sparse matrix or a numpy array")

        if Y is not None:
            if sparse.issparse(Y):
                if not sparse.isspmatrix_csr(Y):
                    Y = sparse.csr_matrix(Y)
            elif isinstance(Y, np.ndarray):
                Y = sparse.csr_matrix(Y)
            else:
                raise ValueError("Y must be a scipy sparse matrix or a numpy array")
            Y.data[Y.data != 1] = 1
            return A, Y
        else:
            return A
