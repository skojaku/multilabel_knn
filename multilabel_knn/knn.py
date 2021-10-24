"""k-nearest neighbor predictor"""
import numpy as np
from scipy import sparse
import faiss


class kNN:
    def __init__(self, k=5, metric="euclidean", exact=False, gpu_id=None):
        self.k = k
        self.metric = metric
        self.gpu_id = gpu_id
        self.exact = exact

    # fit knn model
    def fit(self, X, Y):
        """Fit the model using X as training data and Y as target values

        :param X: training data
        :type X: numpy.ndarray
        :param Y: target values
        :type Y: numpy.ndarray
        :return: self
        :rtype: object
        """
        # make knn graph
        X = self._homogenize(X)
        self.n_indexed_samples = X.shape[0]
        self.index = self._make_faiss_index(X)
        self.Y = Y
        return self

    def predict(self, X):
        """Predict the class labels for the provided data

        :param X: data to predict
        :type X: numpy.ndarray
        :return: predicted class labels
        :rtype: numpy.ndarray
        """

        X = self._homogenize(X)

        A = self._make_knn_graph(X, self.k, exclude_selfloop=False)

        C = A @ self.Y
        cids = np.array(np.argmax(C, axis=1)).reshape(-1)
        Ypred = sparse.csr_matrix(
            (np.ones_like(cids), (np.arange(len(cids)), cids)),
            shape=(len(cids), self.Y.shape[1]),
        )
        return Ypred

    def _make_faiss_index(self, X):
        """Create an index for the provided data

        :param X: data to index
        :type X: numpy.ndarray
        :raises NotImplementedError: if the metric is not implemented
        :return: faiss index
        :rtype: faiss.Index
        """
        n_samples, n_features = X.shape[0], X.shape[1]

        if n_samples < 1000:
            self.exact = True

        if self.metric == "euclidean":
            if self.exact:
                index = faiss.IndexFlatL2(n_features)
            else:
                quantiser = faiss.IndexFlatL2(n_features)

                nlist = int(np.ceil(np.sqrt(n_samples)))
                index = faiss.IndexIVFFlat(
                    quantiser, n_features, nlist, faiss.METRIC_L2
                )
        elif self.metric == "cosine":
            if self.exact:
                index = faiss.IndexFlatIP(n_features)
            else:
                quantiser = faiss.IndexFlatIP(n_features)
                nlist = int(np.ceil(np.sqrt(n_samples)))
                index = faiss.IndexIVFFlat(
                    quantiser, n_features, nlist, faiss.METRIC_INNER_PRODUCT
                )
        else:
            raise NotImplementedError("does not support metric: {}".format(self.metric))

        if self.gpu_id is not None:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)

        if self.exact:
            index.add(X)
        else:
            index.train(X)
            index.add(X)
        return index

    def _make_knn_graph(self, X, k, exclude_selfloop=True):
        """ Construct the k-nearest neighbor graph

        :param X: data to construct the graph
        :type X: numpy.ndarray
        :param k: number of neighbors
        :type k: int
        :param exclude_selfloop: whether to exclude self-loops, defaults to True
        :type exclude_selfloop: bool, optional
        :return: k-nearest neighbor graph
        :rtype: numpy.ndarray
        """
        # get the number of samples and features
        n_samples, n_features = X.shape

        # create a list of k nearest neighbors for each vector
        _, indices = self.index.search(X.astype(np.float32), k)

        rows = np.arange(n_samples).reshape((-1, 1)) @ np.ones((1, k))

        # create the knn graph
        rows, indices = rows.ravel(), indices.ravel()
        if exclude_selfloop:
            s = rows != indices
            rows, indices = rows[s], indices[s]

        A = sparse.csr_matrix(
            (np.ones_like(rows), (rows, indices)),
            shape=(n_samples, self.n_indexed_samples),
        )
        return A

    def _homogenize(self, X, Y=None):
        if self.metric == "cosine":
            X = np.einsum("ij,i->ij", X, 1 / np.linalg.norm(X, axis=1))
        X = X.astype(np.float32)

        if Y is not None:
            if sparse.issparse(Y):
                if not sparse.isspmatrix_csr(Y):
                    Y = sparse.csr_matrix(Y)
            elif isinstance(Y, np.ndarray):
                Y = sparse.csr_matrix(Y)
            else:
                raise ValueError("Y must be a scipy sparse matrix or a numpy array")
            Y.data[Y.data != 1] = 1
            return X, Y
        else:
            return X
