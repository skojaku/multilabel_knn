import shutil
import unittest

import numpy as np
from scipy import sparse

import multilabel_knn as ml


class TestCalc(unittest.TestCase):
    def setUp(self):
        self.X = np.random.randn(300, 30)
        self.Y = sparse.random(self.X.shape[0], 20, density=0.1, format="csr")

    def test_binom_multilabel_knn(self):
        model = ml.binom_multilabel_kNN()
        model.fit(self.X, self.Y)
        model.predict(self.X)

    def test_knn(self):
        model = ml.kNN()
        model.fit(self.X, self.Y)
        model.predict(self.X)

    def test_multilabel_knn(self):
        model = ml.multilabel_kNN()
        model.fit(self.X, self.Y)
        model.predict(self.X)

    def test_binom_multilabel_graph(self):
        model = ml.binom_multilabel_graph()
        A = sparse.random(self.X.shape[0], self.X.shape[0], density=0.1, format="csr")
        model.fit(A, self.Y)
        model.predict(A)


if __name__ == "__main__":
    unittest.main()
