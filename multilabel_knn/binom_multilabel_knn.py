import numpy as np
from scipy import sparse
from numba import njit
from scipy import stats
from .knn import kNN
import faiss
from scipy.special import expit

# class binom_multilabel_kNN:
class binom_multilabel_kNN(kNN):
    """Multilabel k-NN for binomial features.

    This is a modified version of the Multilabel k-NN [1].

    The original multilabel k-nn calculates the posterior probability
    based on the empirical counts of events. An issue of this approach is that
    the empirical counts are crude if the number of samples and labels are large, where
    most rare events never occur.

    To address this problem, we estimte the count distribution using a parametric distribution, i.e., Binomial distribution.
    Because the binomial distribution can be estimated well even with a handful of samples,
    this modified multilabel k-NN can perform well in case the number of samples and labels are large.

    Reference:

    [1] Zhang, Min-Ling, and Zhi-Hua Zhou. 2007.
        “ML-KNN: A Lazy Learning Approach to Multi-Label Learning.”
        Pattern Recognition 40 (7): 2038–48.
    """

    def __init__(
        self,
        k=5,
        metric="euclidean",
        exact=True,
        alpha=1,
        beta=1,
        prior="sample",
        gpu_id=None,
        **params
    ):
        """
        :params n_neighbors: number of neighbors
        :params alpha: hyperparameter for the prior
        :params beta: hyperparameter for the prior
        :param k: [description], defaults to 5
        """
        # super().__init__(k=k, metric=metric, exact=exact, **params)
        self.k = k
        self.metric = metric
        self.gpu_id = gpu_id
        self.exact = exact

        self.prior_type = prior
        self.alpha = alpha
        self.beta = beta

    # fit knn model
    def fit(self, X, Y):
        """Fit the model using X as training data and Y as target values

        :param X: training data
        :type X: numpy.ndarray
        :param Y: target values
        :type Y: numpy.ndarray or scipy.sparse.csr_matrix
        :return: self
        :rtype: object
        """

        X, Y = self._homogenize(X, Y)
        self.Y = Y

        self.n_indexed_samples, self.n_labels = Y.shape[0], Y.shape[1]

        #
        # Calculate the posterior probabilities
        #
        # make knn graph
        self._make_faiss_index(X)
        A = self._make_knn_graph(X, int(self.k), exclude_selfloop=False)
        #
        # Calculate the prior probabilities
        #
        if self.prior_type == "sample":
            self.priors = (self.alpha + np.array(Y.sum(axis=0)).reshape(-1)) / (
                self.alpha + self.beta + self.n_indexed_samples
            )
        elif self.prior_type == "uniform":
            self.priors = np.ones(self.n_labels) * 0.5
        self.p1, self.p0 = self._estimate_binomial_params(A, Y)
        return self

    def predict(self, X, return_prob=False):
        """Predict the target values for X.

        :param X: data to predict
        :type X: numpy.ndarray
        :return: predicted target values
        :rtype: numpy.ndarray
        """

        X = self._homogenize(X)

        A = self._make_knn_graph(X, self.k, exclude_selfloop=False)

        C = A @ self.Y
        samples, labels, count = sparse.find(C)
        count = count.astype(int)

        # Calculate the posterior probabilities
        safelog = lambda x: np.log(np.maximum(x, 1e-12))

        log_likelihood_1 = safelog(self.priors[labels])
        log_likelihood_1 += count * safelog(self.p1[labels]) + (
            self.k - count
        ) * safelog(1 - self.p1[labels])

        log_likelihood_0 = safelog(1 - self.priors[labels])
        log_likelihood_0 += count * safelog(self.p0[labels]) + (
            self.k - count
        ) * safelog(1 - self.p0[labels])

        pred_positive = log_likelihood_1 > log_likelihood_0
        Ypred = sparse.csr_matrix(
            (pred_positive, (samples, labels)), shape=(X.shape[0], self.n_labels)
        )
        if return_prob:
            prob = expit(-log_likelihood_0 + log_likelihood_1)
            Yprob = sparse.csr_matrix(
                (prob, (samples, labels)), shape=(X.shape[0], self.n_labels)
            )
            return Ypred, Yprob
        else:
            return Ypred

    def _estimate_binomial_params(self, A, Y):
        """
        Calculate the conditional probability p for the binomial distribution.

        params: A: knn graph
        params: Y: label matrix
        return: p1, p0: conditional probability for the binomial distribution
        """
        n_nodes, n_labels = Y.shape[0], Y.shape[1]

        Y.sort_indices()
        C1, C0, B1, B0 = _count_neighbors(
            A.indptr, A.indices, Y.indptr, Y.indices, n_nodes, n_labels
        )
        # B1 = self.k * np.array(Y.sum(axis=0)).reshape(-1)
        # Ball = self.k * n_nodes

        p1 = (self.alpha + C1) / (self.alpha + self.beta + B1)
        p0 = (self.alpha + C0) / (self.alpha + self.beta + B0)
        return p1, p0


@njit(nogil=True)
def _isin_sorted(a, x):
    idx = np.searchsorted(a, x)
    if idx >= len(a):
        return False
    else:
        return a[idx] == x


@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    return indices_or_data[indptr[t] : indptr[t + 1]]


@njit(nogil=True)
def _count_neighbors(A_indptr, A_indices, Y_indptr, Y_indices, n_nodes, n_labels):
    C1 = np.zeros(n_labels, dtype=np.int64)
    C0 = np.zeros(n_labels, dtype=np.int64)
    B1 = np.zeros(n_labels, dtype=np.int64)
    B0 = np.zeros(n_labels, dtype=np.int64)
    cnt1 = np.zeros(n_labels, dtype=np.int64)
    cnt0 = np.zeros(n_labels, dtype=np.int64)
    for i in range(n_nodes):
        Y_neighbors_i = _neighbors(Y_indptr, Y_indices, i)
        k = 0
        for j in _neighbors(A_indptr, A_indices, i):  #
            k += 1
            for yj in _neighbors(Y_indptr, Y_indices, j):
                if _isin_sorted(Y_neighbors_i, yj):
                    C1[yj] += 1
                    cnt1[yj] = 1
                else:
                    C0[yj] += 1
                    cnt0[yj] = 1
        B1 += cnt1 * k
        B0 += cnt0 * k
        cnt1 *= 0
        cnt0 *= 0
    return C1, C0, B1, B0
