import abc
from typing import Any

import numpy as np
from sklearn.neighbors import KDTree


class BaseIndex(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build(self, x: np.ndarray, **kwargs: dict[str, Any]) -> None:
        """Builds the index.

        Args:
            x: with shape (n_samples, n_features). Used to build the
                index.
            **kwargs: Additional arguments passed to the index.
        """
        pass

    @abc.abstractmethod
    def query_knn(self, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the (indices, distances) of the k nearest neighbors of v.

        Args:
            v: with shape (n_samples, n_features)
            k: number of neighbors to return

        Returns:
            A tuple of (indices, distances) of the k nearest neighbors of v.
        """
        pass


class KDTreeIndex(BaseIndex):
    def build(self, x: np.ndarray, **kwargs: dict[str, Any]) -> None:
        self.index = KDTree(x, **kwargs)

    def query_knn(self, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        distances, indices = self.index.query([v], k=k)
        return indices[0], distances[0][0]


class AnnoyIndex(BaseIndex):
    def __init__(self, vector_length: int, metric: str = "euclidean") -> None:
        import annoy

        super().__init__()

        self.vector_length = vector_length
        self.metric = metric
        self.index = annoy.AnnoyIndex(self.vector_length, self.metric)

    def build(self, x: np.ndarray, **kwargs: dict[str, Any]) -> None:

        n_trees = 50

        for i, v in zip(range(len(x)), x):
            self.index.add_item(i, v)

        self.index.build(n_trees)

    def query_knn(self, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        indices, distances = self.index.get_nns_by_vector(
            v, k, include_distances=True
        )
        return indices, distances


def create_index(index: str, metric: str, vector_length: int) -> BaseIndex:
    """
    returns the corresponding index based on the passed string

    Args:
        index (string) : name of the index to be created
        metric (string) : distance metric to be used in the index
        vector_length (int) : the feature length, required for the creation of
            the index.
    """
    if index == "annoy":
        return AnnoyIndex(vector_length, metric)
    elif index == "kd_tree":
        return KDTreeIndex()
    else:
        raise ValueError(f"Index {index} not supported")
